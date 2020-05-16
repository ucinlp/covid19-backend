"""
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS train_nli.py {ARGS}
"""
import argparse
import json
import logging
import os
from pathlib import Path

from apex import amp
from apex.parallel import DistributedDataParallel
import transformers
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler
from tqdm import tqdm

from backend.ml.sentence_bert import SentenceBertClassifier

logger = logging.getLogger(__name__)


LABEL_TO_IDX = {
    'entailment': 0,
    'contradiction': 1,
    'neutral': 2
}


class NliDataset(Dataset):

    def __init__(self, fname):
        self._fname = fname

        premises, hypotheses, labels = self._read(fname)

        self._premises = premises
        self._hypotheses = hypotheses
        self._labels = labels

    def _read(self, fname):
        premises = []
        hypotheses = []
        labels = []
        with open(fname, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data['gold_label'] not in LABEL_TO_IDX:
                    continue
                premises.append(data['sentence1'])
                hypotheses.append(data['sentence2'])
                labels.append(data['gold_label'])
        return premises, hypotheses, labels

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        premise = self._premises[idx]
        hypothesis = self._hypotheses[idx]
        label = self._labels[idx]
        return premise, hypothesis, label


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

    parser.add_argument('--fp16', action='store_true')
    # Automatically supplied by torch.distributed.launch
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Additional janky distributed stuff
    args.distributed = False
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    args.distributed = world_size > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info('Loading training data')
    train_dataset = NliDataset(args.train)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=get_sampler(train_dataset, world_size, args.local_rank)
    )

    logger.info('Loading dev data')
    dev_dataset = NliDataset(args.dev)
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
        for premises, hypotheses, labels in iterable:
            optimizer.zero_grad()
            logits = model(premises, hypotheses)
            _, preds = logits.max(dim=-1)
            labels = torch.tensor([LABEL_TO_IDX[l] for l in labels]).cuda()
            acc = (preds == labels).float().mean()
            loss = loss_fn(logits, labels)
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if args.local_rank == 0:
                iterable.set_description(f'Loss: {loss : 0.4f} - Acc: {acc : 0.4f}')
            optimizer.step()

        logger.info('Evaluating...')
        model.eval()
        correct = 0.
        total = 0.
        if args.local_rank == 0:
            iterable = tqdm(dev_dataloader)
        else:
            iterable = dev_dataloader
        for premises, hypotheses, labels in iterable:
            with torch.no_grad():
                logits = model(premises, hypotheses)
            _, preds = logits.max(dim=-1)
            labels = torch.tensor([LABEL_TO_IDX[l] for l in labels]).cuda()
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
