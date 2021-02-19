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
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--dev', type=Path, required=True)
    parser.add_argument('--model-name', type=str, required=True)

    parser.add_argument('--ckpt', type=str, default='ckpt')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('-a', '--accumulation_steps', type=int, default=1)

    parser.add_argument('--fp16', action='store_true')
    # Automatically supplied by torch.distributed.launch
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    # Based on SciFact transoformers training
    class SciFactLabelPredictionDataset(Dataset):
        
        def __init__(self, corpus, claims_file):
            
            claims, rationales, labels = self._read(corpus, claims_file)
          
            self._claims = claims
            self._rationales = rationales
            self._labels = labels
    
        def _read(self, corpus, claims_file):
            claims = []
            rationales = []
            labels = []
            
            corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
            #label_encodings = {'CONTRADICT': 0, 'NOT_ENOUGH_INFO': 1, 'SUPPORT': 2} #From SciFact
            label_encodings = {'CONTRADICT': 1, 'NOT_ENOUGH_INFO': 2, 'SUPPORT': 0} # To Match COVIDLies
    
            for claim in jsonlines.open(claims_file):
               
                if claim['evidence']:
                    for doc_id, evidence_sets in claim['evidence'].items():
                        doc = corpus[int(doc_id)]
    
                        # Add individual evidence set as samples:
                        for evidence_set in evidence_sets:
                            rationale = [doc['abstract'][i].strip() for i in evidence_set['sentences']]
                            claims.append(claim['claim'])
                            rationales.append(' '.join(rationale))
                            labels.append(label_encodings[evidence_set['label']])
    
                        # Add all evidence sets as positive samples
                        rationale_idx = {s for es in evidence_sets for s in es['sentences']}
                        rationale_sentences = [doc['abstract'][i].strip() for i in sorted(list(rationale_idx))]
                        claims.append(claim['claim'])
                        rationales.append(' '.join(rationale_sentences))
                        labels.append(label_encodings[evidence_sets[0]['label']])  # directly use the first evidence set label
                            # because currently all evidence sets have
                            # the same label
                        # Add negative samples
                        non_rationale_idx = set(range(len(doc['abstract']))) - rationale_idx
                        non_rationale_idx = random.sample(non_rationale_idx,
                                                          k=min(random.randint(1, 2), len(non_rationale_idx)))
                        non_rationale_sentences = [doc['abstract'][i].strip() for i in sorted(list(non_rationale_idx))]
                        claims.append(claim['claim'])
                        rationales.append(' '.join(non_rationale_sentences))
                        labels.append(label_encodings['NOT_ENOUGH_INFO'])

                else:
                    # Add negative samples
                    for doc_id in claim['cited_doc_ids']:
                        doc = corpus[int(doc_id)]
                        non_rationale_idx = random.sample(range(len(doc['abstract'])), k=random.randint(1, 2))
                        non_rationale_sentences = [doc['abstract'][i].strip() for i in non_rationale_idx]
                        claims.append(claim['claim'])
                        rationales.append(' '.join(non_rationale_sentences))
                        labels.append(label_encodings['NOT_ENOUGH_INFO'])  

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
    train_dataset = SciFactLabelPredictionDataset(args.corpus, args.train)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=get_sampler(train_dataset, world_size, args.local_rank)
    )

    logger.info('Loading dev data')
    dev_dataset = SciFactLabelPredictionDataset(args.corpus, args.dev)
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