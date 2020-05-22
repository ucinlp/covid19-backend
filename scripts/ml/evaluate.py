import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
from pathlib import Path
import os
import pandas as pd
from collections import OrderedDict

from backend.ml.misconception import MisconceptionDataset
from backend.ml.sentence_bert import SentenceBertClassifier

class AnnotatedDataset(Dataset):

    def __init__(self, fname):
        self._fname = fname

        posts, misinfos, labels = self._read(fname)

        self._posts = posts
        self._misinfos = misinfos
        self._labels = labels

    def _read(self, fname):

        annotations = pd.read_csv (fname)
        label_to_idx = {
        'pos': 0,
        'neg': 1,
        'na': 2 }

        # Filter
        mask = (annotations['tweet'].str.len() < 500) & (annotations['misconception'].str.len() < 500) & (annotations['label'].isin(label_to_idx))
        annotations = annotations.loc[mask]

        # To List
        posts = annotations['tweet'].to_list()
        misinfos = annotations['misconception'].to_list()        
        labels = annotations['label'].to_list()
        labels = [label_to_idx[l] for l in labels]

        return posts, misinfos, labels

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        post = self._posts[idx]
        misinfo = self._misinfos[idx]
        label = self._labels[idx]
        return post, misinfo, label


def load_model (model_dir, weights):
    """
    Load Fine Tuned SentenceBertClassifier
    
    # Parameters
    model_dir : PATH
        Directory path for trained model
    weights: PATH 
        Path for weights.pt
    
    # Returns
    model : SentenceBertClassifier
    """    

    model = SentenceBertClassifier(model_name = model_dir , num_classes=3)
    state_dict = torch.load(weights, map_location='cpu')

    # remove `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    
    return model

def accuracy (confusion_matrix: torch.Tensor):    
    acc = confusion_matrix.diag().sum()/confusion_matrix.sum()  
    return acc.item()

def precision_recall(confusion_matrix: torch.Tensor):
    
    """
    Compute Macro and Micro Averaged Precision and Recall 
    
    # Parameters
    confusion_matrix: torch.Tensor
        Confusion Matrix 
    # Returns
    precision_macro: Float 
        Macro Averaged Precision
    recall_macro: Float 
        Macro Averaged Recall
    precision_micro: Float
        Micro Averaged Precision
    recall_micro: Float
        Micro Averaged Recall
            
    """
    
    preds_pos = torch.sum(confusion_matrix, dim=0) #tp+fp
    true_pos = confusion_matrix.diag() #tp
    labels_pos = torch.sum(confusion_matrix, dim = 1) #tp+fn
        
    class_precisions =  true_pos/preds_pos   
    class_recalls = true_pos/labels_pos
    
    precision_macro = class_precisions.sum()/len(class_precisions)
    recall_macro = class_recalls.sum()/len(class_recalls)
    
    precision_micro = true_pos.sum()/preds_pos.sum()
    recall_micro = true_pos.sum()/labels_pos.sum()
    
    return precision_macro.item(), recall_macro.item(), precision_micro.item(), recall_micro.item()

def get_metrics (model: SentenceBertClassifier, annotations: AnnotatedDataset, batch_size : int, num_classes=3):
    
    """
    Returns Dictionary of Evaluation Metrics 
    
    # Parameters
    model: SentenceBertClassifier
        NLI classifier
    annotations:  AnnotatedDataset
        Test dataset with tweets, misconceptions, and corresponding labels (pos/neg/na)
    batch_size : int
        Batch sizes for predictions (default:10)
    num_classes : int
        Number of classes in classifier (deault = 3)
        
    # Returns
    metrics : Dict
        Dictionary of the following evaluation metrics - 
        Accuracy,  Precision (Macro Averaged), Recall (Macro Averaged), Precision (Micro Averaged), Recall (Micro Averaged)
            
    """

    test_dataloader = DataLoader(
        annotations,
        batch_size = batch_size,
        shuffle = False )
    iterable = tqdm(test_dataloader, position=0, leave=True)
    
    confusion_matrix = torch.zeros(num_classes, num_classes)
       
    for posts, misinfos, labels in iterable:
        with torch.no_grad():
            logits = model(posts, misinfos)
            _, preds = logits.max(dim=-1) 
            
            for y, y_pred in zip(labels, preds):
                confusion_matrix[y, y_pred] += 1

    acc = accuracy(confusion_matrix)
    precision_macro, recall_macro, precision_micro, recall_micro = precision_recall(confusion_matrix)  
   
    metrics = {'Accuracy' : acc,
              'Precision (Macro Avg.)' :  precision_macro ,
              'Recall (Macro Avg.)' : recall_macro ,
              'Precision (Micro Avg.)' : precision_micro ,
              'Recall (Macro Avg.)' : recall_micro }
            
    return metrics

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--weights',type=Path, required=True)    
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--misinfo', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=10)
    args = parser.parse_args()
    
    with open(args.misinfo, 'r') as f:
        misconceptions = MisconceptionDataset.from_jsonl(f)
    
    annotations = AnnotatedDataset(args.test)
    model = load_model(args.model_dir, args.weights)
    model.eval()
    
    metrics = get_metrics (model, annotations, args.batch_size)
    print (metrics)

if __name__ == '__main__':
    main()

    




