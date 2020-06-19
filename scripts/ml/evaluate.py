import torch
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from backend.ml.misconception import MisconceptionDataset
from backend.ml.sentence_bert import SentenceBertClassifier
from backend.stream.db.table import  Output
from backend.stream.db.util import get_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import select
from sqlalchemy import and_


def confusion_matrix(y, y_hat):
    
    confusion_matrix = torch.zeros(3,3)
            
    for t, p in zip(y, y_hat):
        confusion_matrix[t, p] += 1
        
    # Print Confusion Matrix
    row_labels = ['True: Pos', 'True: Neg', 'True: N/A']
    column_labels = ['Pred: Pos', 'Pred: Neg', 'Pred: N/A']
    df = pd.DataFrame(confusion_matrix.numpy(), columns=column_labels, index=row_labels)
    print (df)
        
    return confusion_matrix

def accuracy (confusion_matrix: torch.Tensor):    
    acc = confusion_matrix.diag().sum()/confusion_matrix.sum()  
        
    return acc.item()


def precision_recall_f1(confusion_matrix: torch.Tensor, beta =2):
    
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
        
    class_precision =  true_pos/preds_pos  
    class_recall = true_pos/labels_pos
    class_f1 = beta*(class_precision*class_recall)/(class_precision+class_recall)
             
    return class_precision.tolist(), class_recall.tolist(), class_f1.tolist()


def get_pred_ranks (db,model):
  input_ids = []
  labels = []
  ranks = []

  engine = get_engine(db)
  connection = engine.connect()
  annotated = get_outputs(engine, connection, 'Arjuna')

  for a in annotated:
    if ( a.label_id == 0 or a.label_id == 1) :
        preds = get_outputs(engine, connection, model, a.input_id)
        misinfo_ids = []
        pos = [] # logit for positive prediction
        neg = [] # logit for negative prediction
        
        for p in preds:
            misinfo_ids.append (p.misinfo_id)
            pos.append (p.misc[0])            
            neg.append (p.misc[1])
        
        # If true label is positive, then sort by logit for positive prediction
        # Otherwise sort by logit for negative prediction
        if a.label_id == 0:
            scores = pos
        else:
            scores = neg
                
        zipped = list(zip(misinfo_ids, scores))
        sort = sorted(zipped, key=lambda x: x[1], reverse = True)
        
        # Find rank of misinformation in annotated data        
        rank = [y[0] for y in sort].index(a.misinfo_id) + 1 
        ranks.append (rank)
        labels.append (a.label_id)
        input_ids.append (a.input_id)
            
  connection.close() 
  rank_df = pd.DataFrame ({ 'input_ids' : input_ids,
                            'labels' : labels , 
                            'ranks' : ranks,
                            'inv_ranks' : [1/r for r in ranks] })

  return rank_df


def hits_k(df,k):
    hits = df[df.ranks <= k].shape[0]
    total = df.shape[0]
    
    return hits/total

def hits_k_range(df, n):
  k = []
  overall = []
  pos = []
  neg = []
  for i in range(n):
    k.append(i+1)
    overall.append ( hits_k(df, i+1) ) # Overall

    for j in range(2): # By Class
        cls_df = rank_df[rank_df['labels'] == j]
        if j == 0:
          pos.append (hits_k(cls_df, i+1))
        else :
          neg.append (hits_k(cls_df, i+1))

  hits_df = pd.DataFrame({'k' : k,
                          'Overall' : overall,
                          'Positive' : pos,
                          'Negative' : neg })

   
        
  return hits_df

# Add more cases for possible combinations of inputs
def get_outputs(engine, connection, model=None, input=None, misinfo=None):
    Output.metadata.create_all(bind=engine, checkfirst=True)
    results = None
    try:
        if model is not None:
            if input is not None:
                if misinfo is not None:
                    results = connection.execute(select([Output]).where(and_(Output.model_id == model, Output.input_id == input, Output.misinfo_id == misinfo)))
                else: 
                     results = connection.execute(select([Output]).where(and_(Output.model_id == model, Output.input_id == input)))
            else:
                 results = connection.execute(select([Output]).where(Output.model_id == model))
        else:
            results = connection.execute(select([Output]))
    except SQLAlchemyError as e:
        print(e)
    return results

####################################
########### CMD OPTIONS ############
####################################
parser = argparse.ArgumentParser()
parser.add_argument('--db', type=str, required=True)
parser.add_argument('--model', type=str, required=True)       
args = parser.parse_args()

####################################
###### Load Annotated Data #########
####################################
engine = get_engine(args.db)
connection = engine.connect()
annotated = get_outputs(engine, connection, 'Arjuna')

####################################
###### Load Predictions ############
####################################
y = []
y_hat = []
for a in annotated:
        preds = get_outputs(engine, connection, args.model , a.input_id, a.misinfo_id)
        y.append (a.label_id)
        for p in preds:
            y_hat.append(p.label_id)
   
connection.close()

##################################
######## Compute Metrics #########
##################################
# Confusion Matrix
cm = confusion_matrix(y,y_hat)

##########################
####### Accuracy #########
##########################
acc = accuracy (cm)
print ('Accuracy : ', round(acc*100,3), '%')

###########################
## Precision, Recall, F1 ##
###########################
class_precision, class_recall, class_f1 = precision_recall_f1(cm)

# Print
print ('--- Positive ---')
print ('Precision : ', round(class_precision[0]*100,3), '%' )
print ('Recall : ', round(class_recall[0]*100,3), '%' )
print ('F1 : ', round(class_f1[0]*100,3), '%', '\n' )
print ('--- Negative ---')
print ('Precision : ', round(class_precision[1]*100,3), '%' )
print ('Recall : ', round(class_recall[1]*100,3), '%' )
print ('F1 : ', round(class_f1[1]*100,3), '%', '\n' )
print ('--- Negative ---')
print ('Precision : ', round(class_precision[2]*100,3), '%' )
print ('Recall : ', round(class_recall[2]*100,3), '%' )
print ('F1 : ', round(class_f1[2]*100,3), '%', '\n' )

##########################
##### Rank Metrics  ######
##########################
# Get Ranks
rank_df = get_pred_ranks(args.db, args.model)

# MRR
mrr = np.mean(rank_df.inv_ranks)
print ('MRR : ', mrr)

for i in range(2):
        df = rank_df[rank_df['labels'] == i]
        mrr = np.mean(df.inv_ranks)
        if i == 0:
          cls = 'Positive'
        else:
          cls = 'Negative'        
        print ('MRR for class =', cls, ' : ', mrr)
        

# Hits at K
hits = hits_k_range(rank_df, 10)
print (hits)