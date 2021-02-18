"""
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS train_nli.py {ARGS}
"""
import argparse
import jsonlines
import logging
import os
from pathlib import Path
import random

import transformers
import torch
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler
from tqdm import tqdm

from backend.ml.sentence_bert import SentenceBertClassifier

logger = logging.getLogger(__name__)


def get_sampler(dataset, world_size, rank):
    if world_size <= 1:
        return RandomSampler(dataset)
    else:
        return DistributedSampler(dataset, world_size, rank)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--dev', type=Path, required=True)
    parser.add_argument('--model-name', type=str, required=True)

    parser.add_argument('--ckpt', type=str, default='ckpt')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('-a', '--accumulation_steps', type=int, default=1)
    parser.add_argument('--neg_sample', type=bool, default=False)

    parser.add_argument('--fp16', action='store_true')
    # Automatically supplied by torch.distributed.launch
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    # Based on code from SciFact : 
    class FeverLabelPredictionDataset(Dataset):
        def __init__(self, file):
            
            claims, rationales, labels = self._read(file)
          
            self._claims = claims
            self._rationales = rationales
            self._labels = labels
            
        def _read(self, file):
            claims = []
            rationales = []
            labels = []
            #labels = {'SUPPORTS': 2, 'NOT ENOUGH INFO': 1, 'REFUTES': 0} # From SciFact
            
            label_idx = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2} # To Match COVIDLies
            for data in jsonlines.open(file):
                if data['label'] == 'NOT ENOUGH INFO':
                    if data['sentences']:
                        indices = sorted(random.sample(range(len(data['sentences'])), k=1))
                        sentences = [data['sentences'][i] for i in indices]
                        claims.append(data['claim'])
                        rationales.append(' '.join(sentences))
                        labels.append(label_idx['NOT ENOUGH INFO'])
    
                else:
                    for evidence_set in data['evidence_sets']:
                        claims.append(data['claim'])
                        rationales.append(' '.join([data['sentences'][i] for i in evidence_set]))
                        labels.append(label_idx[data['label']])
                
                    if args.neg_sample:
                        # Add negative samples
                        non_evidence_indices = set(range(len(data['sentences']))) - set(
                            s for es in data['evidence_sets'] for s in es)
                        if non_evidence_indices:
                            non_evidence_indices = random.sample(non_evidence_indices,
                                                                 k=random.randint(1, min(1, len(non_evidence_indices))))
                            sentences = [data['sentences'][i] for i in non_evidence_indices]
                            claims.append(data['claim'])
                            rationales.append(' '.join(sentences))
                            labels.append(label_idx['NOT ENOUGH INFO'])
                            
            return claims, rationales, labels
    

        def __len__(self):
            return len(self._labels)
    
        def __getitem__(self, index):
            claim = self._claims[index]
            rationale = self._rationales[index]
            label = self._labels[index]
            return claim, rationale, label


    # Additional janky distributed stuff
    args.distributed = False
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    args.distributed = world_size > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info('Loading training data')
    train_dataset = FeverLabelPredictionDataset(args.train)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=get_sampler(train_dataset, world_size, args.local_rank)
    )

    logger.info('Loading dev data')
    dev_dataset = FeverLabelPredictionDataset(args.dev)
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        sampler=get_sampler(dev_dataset, world_size, args.local_rank),
        shuffle=False  # Seems weird but the HuggingFace guys do it so...
    )
    

    model = SentenceBertClassifier(model_name=args.model_name, num_classes=3).cuda()
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr)
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    if args.distributed:
        model = DistributedDataParallel(model)
    loss_fn = torch.nn.CrossEntropyLoss()  # Do we need to ignore padding?

    for epoch in range(args.epochs):
        logger.info(f'Epoch: {epoch}')

        logger.info('Training...')
        model.train()
        if args.local_rank == 0:
            iterable = tqdm(train_dataloader)
        else:
            iterable = train_dataloader
        for i, (claims, rationales, labels) in enumerate(iterable):
            if not i % args.accumulation_steps:
                optimizer.step()
                optimizer.zero_grad()

            logits = model(claims, rationales)
            _, preds = logits.max(dim=-1)
            labels = torch.tensor(labels).cuda()
            acc = (preds == labels).float().mean()
            loss = loss_fn(logits, labels)
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if args.local_rank == 0:
                iterable.set_description(f'Loss: {loss : 0.4f} - Acc: {acc : 0.4f}')

        logger.info('Evaluating...')
        model.eval()
        correct = 0.
        total = 0.
        if args.local_rank == 0:
            iterable = tqdm(dev_dataloader)
        else:
            iterable = dev_dataloader
        for claims, rationales, labels in iterable:
            with torch.no_grad():
                logits = model(claims, rationales)
            _, preds = logits.max(dim=-1)
            labels = torch.tensor(labels).cuda()
            correct += (preds == labels).float().sum()
            total += labels.size(0)
            if args.local_rank == 0:
                acc = correct / total
                iterable.set_description(f'Accuracy: {acc.item() : 0.4f}')

        logger.info('Saving...')
        if args.local_rank == 0:
            torch.save(model.state_dict(), f'{args.ckpt}-{epoch}.pt')


if __name__ == '__main__':
    main()