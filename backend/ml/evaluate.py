import argparse
from heapq import heapify, heappush, heappushpop
import json

from tqdm import tqdm

from backend.ml.misconception import MisconceptionDataset
from backend.ml.bertscore import BertScoreDetector


class MaxHeap:
    def __init__(self, k):
        self._k = k
        self._heap = []
        heapify(self._heap)

    def push(self, x):
        """Add an element to the heap."""
        if len(self._heap) < self._k:
            heappush(self._heap, x)
        else:
            heappushpop(self._heap, x)

    def view(self):
        return sorted(self._heap, reverse=True)


def generate_sentences(fname,
                       batch_size=10,
                       field='text'):
    with open(fname, 'r') as f:
        batch = []
        for line in f:
            obj = json.loads(line)
            batch.append(obj[field])
            if len(batch) == batch_size:
                yield batch
                batch = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='input data')
    parser.add_argument('--misconceptions', type=str, help='JSONL file containing misconceptions')
    parser.add_argument('--score_type', type=str, choices=('precision', 'recall', 'f1'), default='f1')
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()

    detector = BertScoreDetector('bert-base-uncased', score_type=args.score_type)

    with open(args.misconceptions, 'r') as f:
        misconceptions = MisconceptionDataset.from_jsonl(f)

    top_scoring_tweets = [MaxHeap(args.k) for _ in range(len(misconceptions))]

    for sentences in tqdm(generate_sentences(args.input)):
        scores = detector.score(sentences, misconceptions)
        # Top-k predictions per misconception
        top_k = scores.argsort(axis=0)[::-1, :][:args.k]
        for misconception_idx in range(len(misconceptions)):
            for sentence_idx in top_k[:, misconception_idx]:
                # Add (score, sentence) tuple to heap
                x = (scores[sentence_idx, misconception_idx], sentences[sentence_idx])
                top_scoring_tweets[misconception_idx].push(x)

    for i, misconception in enumerate(misconceptions.sentences):
        print(f'Misconception: {misconception}')
        for score, sentence in top_scoring_tweets[i].view():
            print(f'\t{score} :: {sentence}')


if __name__ == '__main__':
    main()
