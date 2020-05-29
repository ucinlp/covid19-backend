import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
from pathlib import Path
import os
import pandas as pd
import numpy as np
from collections import OrderedDict

from backend.ml.misconception import MisconceptionDataset
from backend.ml.sentence_bert import SentenceBertClassifier

class AnnotatedDataset(Dataset):

    def __init__(self, fname):
        self._fname = fname

        posts, misinfos, ids, labels = self._read(fname)

        self._posts = posts
        self._misinfos = misinfos
        self._labels = labels
        self._ids = ids

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
        ids = annotations['misconception_id'].to_list()  
        labels = annotations['label'].to_list()
        labels = [label_to_idx[l] for l in labels]

        return posts, misinfos, ids, labels

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        post = self._posts[idx]
        misinfo = self._misinfos[idx]
        label = self._labels[idx]
        id = self._ids[idx]
        return post, misinfo, id, label 


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
    state_dict = torch.load(weights)

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
    class_precisions: Float 
       Precision for each class
    class_recalls: Float 
       Recall for each class

    """
    
    preds_pos = torch.sum(confusion_matrix, dim=0) #tp+fp
    true_pos = confusion_matrix.diag() #tp
    labels_pos = torch.sum(confusion_matrix, dim = 1) #tp+fn
        
    class_precisions =  true_pos/preds_pos   
    class_recalls = true_pos/labels_pos
      
    return class_precisions.tolist(), class_recalls.tolist()

def hits_k(df,k):
    hits = df[df.ranks <= k].shape[0]
    total = df.shape[0]
    
    return hits/total
    
def cm_metrics (model: SentenceBertClassifier, annotations: AnnotatedDataset, batch_size : int, num_classes=3):
    
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
        
    # Printss
    acc:
        Accuracy    
    class_precisions: Float 
       Precision for each class
    class_recalls: Float 
       Recall for each class
            
    """

    test_dataloader = DataLoader(
        annotations,
        batch_size = batch_size,
        shuffle = False )
    iterable = tqdm(test_dataloader, position=0, leave=True)
    
    confusion_matrix = torch.zeros(num_classes, num_classes)
       
    for posts, misinfos, id, labels in iterable:
        with torch.no_grad():
            logits = model(posts, misinfos)
            _, preds = logits.max(dim=-1) 
            
            for y, y_pred in zip(labels, preds):
                confusion_matrix[y, y_pred] += 1

    acc = accuracy(confusion_matrix)
    class_precisions, class_recalls = precision_recall(confusion_matrix)  
   
    print ('Accuracy : ', acc)
    print ('Class Precisions : ', class_precisions)    
    print ('Class Recalls : ', class_recalls)

def rank_metrics (model: SentenceBertClassifier, annotations: AnnotatedDataset, misconceptions: MisconceptionDataset, k):
    
    """
    Returns Dictionary of Evaluation Metrics 
    
    # Parameters
    model: SentenceBertClassifier
        NLI classifier
    annotations:  AnnotatedDataset
        Test dataset with tweets, misconceptions, and corresponding labels (pos/neg/na)
    misconceptions : MisconceptionDataset
    k : int
        Value of K for Hits@k
        
    # Prints
        Hits@k and MRR for Positive and Negative classes
            
    """
    ### Hits at K at MRR    
    ranks = []
    labels = []
    for post, misinfo, id, label in tqdm(annotations):
        scores=[]
        ids = []        
        if ( label == 0 or label == 1) : #Only for Positive or Negative Labels            
            for miscon in misconceptions:           
                with torch.no_grad():
                    logits = model([post], [miscon.pos_variations[0]])
                    logit = logits[0][label]
                    
                    ids.append( int( miscon.id ) )
                    scores.append(logit.item())    
             
            zipped = list(zip(ids, scores))
            sort = sorted(zipped, key=lambda x: x[1], reverse = True)
                
            rank = [y[0] for y in sort].index(id) + 1
            ranks.append (rank)
            labels.append (label)
            
    rank_df = pd.DataFrame ({'labels' : labels , 
                             'ranks' : ranks,
                             'inv_ranks' : [1/r for r in ranks] })
    
    for i in range(2):
        print ('--- CLASS = ', i, '---')
        df = rank_df[rank_df['labels'] == i]
        hk = hits_k (df, k)
        mrr = np.mean(df.inv_ranks)
        print ('Hits at K = ', k, ' : ', hk)
        print ('MRR : ', mrr)
     

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--weights',type=Path, required=True)    
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--misinfo', type=Path, required=True)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=10)
    args = parser.parse_args()
    
    with open(args.misinfo, 'r') as f:
        misconceptions = MisconceptionDataset.from_jsonl(f)
    
    annotations = AnnotatedDataset(args.test)
    model = load_model(args.model_dir, args.weights)
    model.eval()
    
    cm_metrics (model, annotations, args.batch_size)    
    rank_metrics (model, annotations, misconceptions, args.k)

if __name__ == '__main__':
    main()

    




