import ast
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import select
from sqlalchemy import and_

from backend.stream.db.table import Output
from backend.stream.db.util import get_engine


SIMILARITY_MODELS = [
    'bow-cosine',
    'bert-base-cosine',
    'bert-ft-cosine',
    'ct-bert-cosine',
    'bert-score-base',
    'bert-score-ft',
    'bert-score-ct',
    'glove-avg-cosine',
    'pyserini'
]

STANCE_MODELS = [
    'bow-log-snli',
    'bow-log-mnli',
    'bow-log-mednli',
    'boe-log-snli',
    'boe-log-mnli',
    'boe-log-mednli',
    'bilstm-snli',
    'bilstm-mnli',
    'bilstm-mednli',
    'sbert-snli',
    'sbert-mnli',
    'sbert-mednli',
    'sbert-snli-ct',
    'sbert-mnli-ct',
    'sbert-mednli-ct',
    'sbert-fever-ct',
    'sbert-scifact-ct',
    'sbert-ann-ct',
    'comb-bilstm-snli',
    'comb-bilstm-mnli',
    'comb-bilstm-mednli',
    'comb-sbert-snli-ct',
    'comb-sbert-mnli-ct',
    'comb-sbert-mednli-ct',
    'comb-sbert-fever-ct',
    'comb-sbert-scifact-ct',
    'comb-sbert-ann-ct'
]


def confusion_matrix(y, y_hat):
    confusion_matrix = torch.zeros(3, 3)
    for t, p in zip(y, y_hat):
        confusion_matrix[t, p] += 1
    row_labels = ['True: Pos', 'True: Neg', 'True: N/A']
    column_labels = ['Pred: Pos', 'Pred: Neg', 'Pred: N/A']
    df = pd.DataFrame(confusion_matrix.numpy(), columns=column_labels, index=row_labels)
    print(confusion_matrix)
    return confusion_matrix


def accuracy(confusion_matrix: torch.Tensor):    
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
    preds_pos = torch.sum(confusion_matrix, dim=0)  # tp+fp
    true_pos = confusion_matrix.diag()  # tp
    labels_pos = torch.sum(confusion_matrix, dim = 1) # tp+fn

    class_precision =  true_pos/preds_pos  
    class_recall = true_pos/labels_pos
    class_f1 = beta*(class_precision*class_recall)/(class_precision+class_recall)

    return class_precision.tolist(), class_recall.tolist(), class_f1.tolist()


def get_annotated_data(db, annotated_data):
    input_ids = []
    misinfo_ids = []
    labels = []

    engine = get_engine(db)
    connection = engine.connect()
    annotated = get_outputs(engine, connection, annotated_data)

    for a in annotated:
        input_ids.append (a.input_id)
        misinfo_ids.append (a.misinfo_id)
        labels.append (a.label_id)     

    connection.close()

    annotated = pd.DataFrame({
        'input_id': input_ids,
        'gold_label': labels,
        'misinfo_id': misinfo_ids,
    })  
    
    annotated['gold_label'] = annotated.gold_label.astype(int)

    return annotated


def get_preds_db(db, model_name):    
    input_id = []
    labels = []
    mid = []
    confidence = []

    if model_name in STANCE_MODELS:
        pos = []
        neg =[]

    engine = get_engine(db)
    connection = engine.connect()
    preds = get_outputs(engine, connection, model_name)

    # Load predictions
    for p in preds:
          input_id.append (p.input_id)
          mid.append (p.misinfo_id)
          labels.append (p.label_id)
          confidence.append (p.confidence)
          
          if model_name in STANCE_MODELS:
              pos.append (p.misc[0])
              neg.append (p.misc[1])

    connection.close()

    # Create data frame
    pred_df = pd.DataFrame({
        'input_id': input_id,
        'model_id': model_name,
        'label_id': labels,
        'misinfo_id': mid,
        'confidence': confidence,
    })
    
    
    if model_name in SIMILARITY_MODELS:
        grouped = pred_df.groupby('input_id')
        ranked_confidence = grouped['confidence'].rank(ascending=False)
        pred_df['rank'] = ranked_confidence
    else:
        pred_df['label_id'] = pred_df.label_id.astype(int)
    return pred_df


def get_preds_file(file_name, model_name):    
    pred_df = pd.read_csv(file_name, converters={'misc':ast.literal_eval})
  
    if model_name in model_name in SIMILARITY_MODELS:
        pred_df['rank'] = pred_df.groupby('input_id')['confidence'].rank(ascending=False) # Ranks
  
    return pred_df


def hits_k(df, k):
    hits = df[df['rank'] <= k].shape[0]
    total = df.shape[0]    
    return round( hits/total*100 , 1 )


def get_outputs(
    engine,
    connection,
    model=None,
    input=None,
    misinfo=None,
):
    Output.metadata.create_all(bind=engine, checkfirst=True)
    results = None
    try:
        if model is not None:
            condition = Output.model_id == model
            if input is not None:
                condition = and_(condition, Output.input_id == input)
            if misinfo is not None:
                condition = and_(condition, Output.model_id == misinfo)
            results = connection.execute(select([Output]).where(condition))
        else:
            results = connection.execute(select([Output]))
    except SQLAlchemyError as e:
        print(e)
    return results


def main(args):
    # Load data
    annotated = get_annotated_data(args.db, args.eval_data)
    # Predictions
    if args.file_name:
        pred_df = get_preds_file(args.file_name, args.model_name)
    else:
        pred_df = get_preds_db(args.db, args.model_name)
    # Combine
    eval_df = annotated.merge(
        pred_df,
        left_on=['input_id', 'misinfo_id'],
        right_on=['input_id', 'misinfo_id'],
        how='left',
    )
    # Compute metrics
    if args.model_name in SIMILARITY_MODELS:
        eval_df = eval_df.assign(inv_rank = lambda x: 1/x['rank'])

        # Only keep positive or negative true labels
        eval_df = eval_df[eval_df['gold_label'].isin([0, 1])]
        pos = eval_df[eval_df['gold_label'] == 0]

        # MRR
        pos_mrr = round(np.mean(pos.inv_rank), 2 )  # Postive 
        both_mrr = round(np.mean(eval_df.inv_rank), 2)  # Positive and Negative

        # Hits @ K

        # Positive
        pos_h_1 = hits_k(pos, 1)
        pos_h_5 = hits_k(pos, 5)
        pos_h_10 = hits_k(pos, 10)

        # Postive and Negative
        both_h_1 = hits_k(eval_df, 1)
        both_h_5 = hits_k(eval_df, 5)
        both_h_10 = hits_k(eval_df, 10)

        output_string = ' & '.join([
            str(pos_h_1),
            str(pos_h_5),
            str(pos_h_10),
            str(pos_mrr),
            str(both_h_1),
            str(both_h_5),
            str(both_h_10),
            str(both_mrr),
        ])

        print(output_string)

    elif args.model_name in STANCE_MODELS:
                
        # Classification Metrics
        cm = confusion_matrix(eval_df.gold_label, eval_df.label_id)
        class_precision, class_recall, class_f1 = precision_recall_f1(cm)
        
        # Postive
        pos_pr = round(class_precision[0]*100, 1)
        pos_re = round(class_recall[0]*100, 1)
        pos_f1 = round(class_f1[0]*100, 1)

        # Negative
        neg_pr = round(class_precision[1]*100, 1)
        neg_re = round(class_recall[1]*100, 1)
        neg_f1 = round(class_f1[1]*100, 1)

        # Neutral 
        na_pr = round(class_precision[2]*100, 1)
        na_re = round(class_recall[2]*100, 1)
        na_f1 = round(class_f1[2]*100, 1)
        
        # Macro
        macro = precision_recall_fscore_support(
            eval_df.gold_label,
            eval_df.label_id,
            average='macro',
        )
        mac_pr = round(macro[0]*100, 1)
        mac_re = round(macro[1]*100, 1)
        mac_f1 = round(macro[2]*100, 1)
        
        # Accuracy
        acc = round(accuracy_score(eval_df.gold_label, eval_df.label_id) * 100, 1)
        print(acc)
        
        # Print Output
        output_string = ' & '.join([
            str(mac_pr),
            str(mac_re),
            str(mac_f1),
            str(pos_pr),
            str(pos_re),
            str(pos_f1),
            str(neg_pr),
            str(neg_re),
            str(neg_f1),
            str(na_pr),
            str(na_re),
            str(na_f1)
        ])
        print(output_string)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--file_name', type=Path, required=False)
    parser.add_argument('--eval_data', type=str, required=True) 
    args = parser.parse_args()

    main(args)

