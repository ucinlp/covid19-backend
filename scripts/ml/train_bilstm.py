import os
import dill
import json
import argparse
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data

from backend.ml.bi_lstm import BiLSTM

def map_values(row, values_dict):
    return values_dict[row]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

values_dict = {'neutral': 2, 
               'entailment': 0, 
               'contradiction': 1, 
               '-' : 3}

def json_to_csv (file, output_file):
  data = []
  with open(file) as f:
    for line in f:
        data.append(json.loads(line))
  df = pd.DataFrame.from_records(data)[['sentence1', 'sentence2', 'gold_label']]
  df['gold_label'] = df['gold_label'].map(values_dict)  
  df = df[df['gold_label'].isin([0, 1, 2])]
  df.to_csv (output_file, index = False)

def accuracy(preds, y):
    max_preds = preds.argmax(dim = 1, keepdim = True)
    correct = max_preds.squeeze(1).eq(y).to(device)
    length =torch.FloatTensor([y.shape[0]]).to(device)
    return correct.sum() / length
  
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--dev', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=512)

    args = parser.parse_args()  
        
    TEXT = data.Field(tokenize = 'spacy', lower = True)
    LABEL = data.LabelField()
    train_csv = os.path.join('/',args.output_dir, 'train.csv')
    json_to_csv(args.train, train_csv)
    dev_csv = os.path.join('/',args.output_dir, 'dev.csv')
    json_to_csv(args.dev, dev_csv)    
    
    train_data, val_data = data.TabularDataset.splits(
            path=args.output_dir, 
            train='train.csv',  
            validation='dev.csv', 
            format='csv', 
            skip_header = True,
            fields=[('sentence1', TEXT), ('sentence2', TEXT), ('gold_label', LABEL)])
    
    TEXT.build_vocab(train_data, 
                 min_freq = 2,
                 vectors = "glove.6B.300d",
                 unk_init = torch.Tensor.normal_)
    field_path = os.path.join('/',args.output_dir, 'bilstm-field.pt')
    torch.save(TEXT, 
               field_path, 
               pickle_module=dill)
    LABEL.build_vocab(train_data)
    

    train_iterator, valid_iterator= data.BucketIterator.splits(
            (train_data, val_data), 
            batch_size = args.batch_size,
            device = device,
            sort_key=lambda x: len(x.sentence1),
            sort_within_batch=False)

    
    pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
    model = BiLSTM(input_dim=len(TEXT.vocab), 
                 embedding_dim=300,
                 hidden_dim=300,
                 lstm_layers=2,
                 fc_layers=3,
                 output_dim=len(LABEL.vocab) , 
                 dropout=0.25, 
                 pad_idx=pad_idx).to(device)
    
    model.embedding.weight.data[pad_idx] = torch.zeros(300)
    model.embedding.weight.requires_grad = True
    optimizer = optim.Adam(model.parameters())
    ce_loss = nn.CrossEntropyLoss().to(device)
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    best_valid_loss = float('inf')
    model_path = os.path.join('/',args.output_dir, 'bilstm.pt') 

    for epoch in range(args.epochs):

        train_loss = 0
        train_acc = 0    
        model.train()
        
        for batch in train_iterator:        
            prem = batch.sentence1
            hypo = batch.sentence2
            labels = batch.gold_label
            
            optimizer.zero_grad()
            predictions = model(prem, hypo)        
            loss = ce_loss(predictions, labels)                
            acc = accuracy(predictions, labels)
            
            loss.backward()        
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += acc.item()
            
        train_loss = train_loss / len(train_iterator)
        train_acc = train_acc / len(train_iterator)
        
        valid_loss = 0
        valid_acc = 0
    
        model.eval()
    
        with torch.no_grad():
        
            for batch in valid_iterator:
    
                prem = batch.sentence1
                hypo = batch.sentence2
                labels = batch.gold_label
                            
                predictions = model(prem, hypo)                
                loss = ce_loss(predictions, labels)                    
                acc = accuracy(predictions, labels)
                
                valid_loss += loss.item()
                valid_acc += acc.item()
        
        valid_loss = valid_loss / len(valid_iterator)
        valid_acc = valid_acc / len(valid_iterator)
    
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_path)
    
        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    

if __name__ == '__main__':
    main()