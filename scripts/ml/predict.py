import os
import dill
import torch
import argparse
import pandas as pd
from pathlib import Path
from pyserini.search import SimpleSearcher
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import select
from sqlalchemy import and_

from backend.stream.db.util import get_engine
from backend.stream.db.operation import get_inputs, put_outputs
from backend.stream.db.table import  Output
from backend.ml.misconception import MisconceptionDataset
from backend.ml.bertscore import BertScoreDetector
from backend.ml.baseline_models import BoWCosine, GloVeCosine, BERTCosine, BoWLogistic, BoELogistic
from backend.ml.bi_lstm import BiLSTM
from backend.ml.sentence_bert import SentenceBertClassifier


if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'
    
def load_sbert_model (model_dir, weights):
    """
    Load SentenceBertClassifier
    
    # Parameters
    model_dir : PATH
        Directory path for trained model
    weights: PATH 
        Path for weights.pt
    
    # Returns
    model : SentenceBertClassifier
    """    
    model = SentenceBertClassifier(model_name = model_dir , num_classes=3)
    state_dict = torch.load(weights, map_location=map_location )
    model.load_state_dict(state_dict,strict=False)    
    return model

def load_bilstm (model_path, tokenizer_path):
    model = torch.load(model_path,  map_location=map_location)
    model.field = torch.load (tokenizer_path, pickle_module=dill, map_location=map_location)
    return model


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

# Classify Agree vs. Disagree for Combined Models
def classify_agree_disagree(row):
    if row['rel_label_id'] == 2:
        val = 2
    elif row['rel_label_id'] == 3 and row['probs'][0] > row['probs'][1]:
        val = 0
    elif row['rel_label_id'] == 3 and row['probs'][1] > row['probs'][0]:
        val = 1
    return val

sm = torch.nn.Softmax(dim=-1)

def main():
       
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--model_dir', type=Path) #Required for entailment models and pyserini retrieval
    parser.add_argument('--db_input', type=str, required=True)
    parser.add_argument('--file', type=Path, required=False)
    parser.add_argument('--db_output', type=str,  required=False)
    
    args = parser.parse_args()
    model_name = args.model_name
    db_input = args.db_input
    
    #Similarity
    if model_name == 'glove-avg-cosine':
        model = GloVeCosine()
    
    if model_name == 'bert-base-cosine':
        model = BERTCosine('base')
        
    if model_name == 'ct-bert-cosine':
        model = BERTCosine('ct-bert')
        
    if model_name == 'bert-score-base':
      model = BertScoreDetector('bert-large-uncased')

    if model_name == 'bert-score-ft':
      model = BertScoreDetector('models/roberta-ckpt/covid-roberta')

    if model_name in ['bert-score-ct','comb-bilstm-snli', 'comb-bilstm-mnli',
                      'comb-bilstm-mednli', 'comb-sbert-snli-ct', 'comb-sbert-mnli-ct',
                      'comb-sbert-mednli-ct', 'comb-sbert-fever-ct', 'comb-sbert-scifact-ct',
                      'comb-sbert-ann-ct']:
      model = BertScoreDetector('digitalepidemiologylab/covid-twitter-bert')
      
    if model_name == 'pyserini':
        model_dir = str(args.model_dir)
        searcher = SimpleSearcher(model_dir)

    #Entailment
    # BoW Logistic
    if model_name in ['bow-log-snli', 'bow-log-mnli', 'bow-log-mednli']:  
        model = BoWLogistic(args.model_dir)
        
    # BoE Logistic
    if model_name in ['boe-log-snli', 'boe-log-mnli', 'boe-log-mednli']:
        model = BoELogistic(args.model_dir)        
     
    if model_name == 'sbert-mnli':
        model_path = os.path.join('/', args.model_dir, 'mnli-sbert-ckpt-1.pt')
        model = load_sbert_model('bert-base-cased', model_path)
        
    if model_name == 'sbert-snli':
        model_path = os.path.join('/', args.model_dir, 'snli-sbert-ckpt-2.pt')
        model = load_sbert_model('bert-base-cased', model_path)
    
    if model_name == 'sbert-mednli':   
        model_path = os.path.join('/', args.model_dir, 'mednli-sbert-ckpt-5.pt')  
        model = load_sbert_model('bert-base-cased', model_path)
    
    if model_name in ['sbert-snli-ct', 'comb-sbert-snli-ct']:   
        model_path = os.path.join('/', args.model_dir, 'snli-sbert-ct-ckpt-2.pt')
        if model_name == 'sbert-snli-ct':
            model = load_sbert_model('digitalepidemiologylab/covid-twitter-bert', model_path)  
        elif model_name == 'comb-sbert-snli-ct':
            model2 = load_sbert_model('digitalepidemiologylab/covid-twitter-bert', model_path)    
    
    if model_name in ['sbert-mnli-ct', 'comb-sbert-mnli-ct']:   
        model_path = os.path.join('/', args.model_dir, 'mnli-sbert-ct-ckpt-2.pt')
        if model_name == 'sbert-mnli-ct':
            model = load_sbert_model('digitalepidemiologylab/covid-twitter-bert', model_path)
        elif model_name == 'comb-sbert-mnli-ct':
            model2 = load_sbert_model('digitalepidemiologylab/covid-twitter-bert', model_path)
    
    if model_name in ['sbert-mednli-ct', 'comb-sbert-mednli-ct'] :   
        model_path = os.path.join('/', args.model_dir, 'mednli-sbert-ct-ckpt-7.pt')
        if model_name == 'sbert-mednli-ct':
            model = load_sbert_model('digitalepidemiologylab/covid-twitter-bert', model_path)    
        elif model_name == 'comb-sbert-mednli-ct':
            model2 = load_sbert_model('digitalepidemiologylab/covid-twitter-bert', model_path)               
            
    if model_name in ['sbert-ann-ct', 'comb-sbert-ann-ct'] :   
        #model_path = os.path.join('/', args.model_dir, 'ann-sbert-ct-9.pt')
        model_path = os.path.join('/', args.model_dir, 'ann-ch-sbert-ct-3.pt')
        if model_name == 'sbert-ann-ct':
            model = load_sbert_model('digitalepidemiologylab/covid-twitter-bert', model_path)    
        elif model_name == 'comb-sbert-ann-ct':
            model2 = load_sbert_model('digitalepidemiologylab/covid-twitter-bert', model_path)
            
    if model_name in ['sbert-fever-ct', 'comb-sbert-fever-ct'] :   
        model_path = os.path.join('/', args.model_dir, 'fever-sbert-ct-1.pt')
        if model_name == 'sbert-fever-ct':
            model = load_sbert_model('digitalepidemiologylab/covid-twitter-bert', model_path)    
        elif model_name == 'comb-sbert-fever-ct':
            model2 = load_sbert_model('digitalepidemiologylab/covid-twitter-bert', model_path)
            
    if model_name in ['sbert-scifact-ct', 'comb-sbert-scifact-ct'] :   
        model_path = os.path.join('/', args.model_dir, 'scifact-sbert-ct-2.pt')
        if model_name == 'sbert-scifact-ct':
            model = load_sbert_model('digitalepidemiologylab/covid-twitter-bert', model_path)    
        elif model_name == 'comb-sbert-scifact-ct':
            model2 = load_sbert_model('digitalepidemiologylab/covid-twitter-bert', model_path)
            
    if model_name in ['bilstm-snli', 'bilstm-mnli', 'bilstm-mednli', 
                      'comb-bilstm-snli', 'comb-bilstm-mnli', 'comb-bilstm-mednli' ]:
        model_path = os.path.join('/', args.model_dir, 'bilstm.pt')
        field_path = os.path.join('/', args.model_dir, 'bilstm-field.pt')
        if model_name in ['bilstm-snli', 'bilstm-mnli', 'bilstm-mednli']:
            model = load_bilstm (model_path, field_path )
        elif model_name in ['comb-bilstm-snli', 'comb-bilstm-mnli', 'comb-bilstm-mednli']:
            model2 = load_bilstm (model_path, field_path )
            
        
    if model_name in ['bert-score-base', 'bert-score-ft', 'bert-score-ct',
                      'bilstm-snli', 'bilstm-mnli','bilstm-mednli', 
                      'sbert-snli', 'sbert-mnli','sbert-mednli',
                      'sbert-snli-ct', 'sbert-mnli-ct','sbert-mednli-ct',
                      'comb-bilstm-snli', 'comb-bilstm-mnli', 'comb-bilstm-mednli', 
                      'comb-sbert-snli-ct','comb-sbert-mnli-ct', 'comb-sbert-mednli-ct',
                      'sbert-fever-ct', 'comb-sbert-fever-ct',
                      'sbert-scifact-ct', 'comb-sbert-scifact-ct'
                      'sbert-ann-ct', 'comb-sbert-ann-ct']:
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        
        if model_name in ['comb-bilstm-snli', 'comb-bilstm-mnli', 'comb-bilstm-mednli', 
                          'comb-sbert-snli-ct', 'comb-sbert-mnli-ct', 'comb-sbert-mednli-ct',
                          'comb-sbert-fever-ct', 'comb-sbert-scifact-ct','comb-sbert-ann-ct']: 
            model2.eval()
            if torch.cuda.is_available():
                model2.cuda()
                
    if model_name in ['bert-base-cosine', 'ct-bert-cosine']:
        model.model.eval()
        if torch.cuda.is_available():
            model.model.cuda()
        
    
    #Read misinformation
    misinfos = MisconceptionDataset.from_db(db_input)    
    mis = []
    mid = []    
    for misinfo in misinfos:
        mis.append(misinfo.pos_variations[0] )
        mid.append(misinfo.id)    
    n = len(mid)    
    # Encode misconception
    if model_name in ['bert-base-cosine', 'bert-ft-cosine', 'ct-bert-cosine', 
                      'bert-score-base', 'bert-score-ft','bert-score-ct',
                      'glove-avg-cosine', 'boe-log-snli', 'boe-log-mnli', 
                      'boe-log-mednli' ,'bilstm-snli', 'bilstm-mnli', 'bilstm-mednli', 
                      'comb-bilstm-snli', 'comb-bilstm-mnli', 'comb-bilstm-mednli',
                      'comb-sbert-snli-ct', 'comb-sbert-mnli-ct', 'comb-sbert-mednli-ct',
                      'comb-sbert-fever-ct', 'comb-sbert-scifact-ct','comb-sbert-ann-ct']:
        mis_vect = model._encode (mis)
    
    elif model_name in ['bow-log-snli', 'bow-log-mnli', 'bow-log-mednli']:
        mx = model._encode (mis, 'hyp')
          
    #Predict
    # Tweets
    engine = get_engine(db_input)
    connection = engine.connect()
    inputs = get_inputs(engine, connection)
    
    output = []
    
    for input in inputs:
        print (input['id'] )
        posts = [input['text']] * n
        post_ids = [input['id']] * n
        
        #Similarity Models
        #BoW Cosine
        if model_name == 'bow-cosine':
          corpus = mis + [input['text']]
          model = BoWCosine(corpus)
          mis_vect = model._encode(mis)      
        
        if model_name in ['bow-cosine', 'bert-base-cosine', 'bert-ft-cosine', 
                          'ct-bert-cosine', 'bert-score-base', 'bert-score-ft',
                          'bert-score-ct','glove-avg-cosine' ]:
            post_vect = model._encode([input['text']])
            score = model._score(post_vect, mis_vect)
            df = pd.DataFrame({'input_id': post_ids, 
                               'model_id': model_name, 
                               'label_id': 'n/a',
                               'misinfo_id': mid, 
                               'confidence': score[0] }).to_dict('records')
    
        if model_name == 'pyserini':
            hits = searcher.search(input['text'], k = 86)
    
            mid2 = []
            score = []
            # Print the first 10 hits:
            for i in range(0, len(hits)):
                mid2.append (int (hits[i].docid) )
                score.append (hits[i].score)
            
        
            missed = set(mid)-set (mid2)
            missed = list( missed )
        
            for m in missed:
                mid2.append(m)
                score.append (0)
             
            df = pd.DataFrame({'input_id': post_ids, 
                               'model_id': model_name, 
                               'label_id': 'n/a',
                               'misinfo_id': mid2, 
                               'confidence': score}).to_dict('records')
        
        #Entailment Models
        #BoW Logistic
        if model_name in ['bow-log-snli', 'bow-log-mnli', 'bow-log-mednli']:
            px = model._encode(posts, 'prem')         
            preds, probs = model._predict(px,mx)    
            max_probs = probs.max(axis =1)            
           
        #Boe Logistic
        if model_name in ['boe-log-snli', 'boe-log-mnli', 'boe-log-mednli']:
            post_vect = model._encode([input['text']]) * n
            
            preds, probs = model._predict(post_vect, mis_vect)    
            max_probs = probs.max(axis =1) 
        
        #BiLSTM
        if model_name in ['bilstm-snli', 'bilstm-mnli', 'bilstm-mednli', 'bilstm-fnc']:
            post_vect = model._encode([input['text']]* n) 
            with torch.no_grad():
                logits = model(post_vect, mis_vect)  
                probs = sm(logits)
                max = probs.max(dim=-1)
                max_probs = max[0].tolist()
                preds = max[1].tolist()
                  
        #SBERT
        if model_name in ['sbert-snli', 'sbert-mnli', 'sbert-mednli', 
                          'sbert-mednli-ct', 'sbert-mnli-ct', 'sbert-snli-ct',
                          'sbert-fever-ct', 'sbert-scifact-ct','sbert-ann-ct']:
            with torch.no_grad():
                logits = model(posts, mis)           
                _, preds = logits.max(dim=-1)                        
                probs = sm(logits)
                max_probs, _ = probs.max(dim=-1)
        
        #Stacked # BiLSTM and SBERT
        if model_name in ['comb-bilstm-snli', 'comb-bilstm-mnli', 'comb-bilstm-mednli', 
                          'comb-sbert-snli-ct', 'comb-sbert-mnli-ct', 'comb-sbert-mednli-ct', 
                          'comb-sbert-fever-ct', 'comb-sbert-scifact-ct','comb-sbert-ann-ct']:
          ## BertScore-DA
          post_vect = model._encode([input['text']])
          score = model._score(post_vect, mis_vect)
          bert_score = score[0]
    
          ## Relevance Classification
          rel_preds = [ 3 if x >= 0.4 else 2 for x in bert_score ]
          df = pd.DataFrame({'input_id': post_ids, 
                             'input': posts,
                             'model_id': model_name, 
                             'rel_label_id': rel_preds,
                             'misinfo' : mis, 
                             'misinfo_id': mid, 
                             'confidence': bert_score})
          
          ## Agree/Disagree Classification
          relevant = df[df.rel_label_id ==3]
          
          if relevant.shape[0] > 0: 
            post = relevant.input.tolist()
            misinfo = relevant.misinfo.tolist()  
              
            if model_name in ['comb-bilstm-snli', 'comb-bilstm-mnli', 'comb-bilstm-mednli']:
                  post = model2._encode(post)
                  misinfo = model2._encode(misinfo)

            with torch.no_grad():
              logits = model2(post, misinfo) 
              probs = sm(logits)
              relevant['probs'] = probs.tolist()
    
              relevant = relevant[['input_id', 'misinfo_id', 'probs']] 
              df = df.merge (relevant, 
                             how = 'left', 
                             left_on = ['input_id', 'misinfo_id'], 
                             right_on = ['input_id', 'misinfo_id'])

              df['label_id'] = df.apply(classify_agree_disagree, axis=1)
    
          else:
            df['label_id'] = df['rel_label_id']
          
          df = df[['input_id', 'model_id', 'label_id', 'misinfo_id', 'confidence']].to_dict('records')
        
        #Output        
        if model_name in ['bow-cosine', 'bert-base-cosine', 'bert-ft-cosine', 
                          'ct-bert-cosine', 'bert-score-base', 
                          'bert-score-ft','bert-score-ct', 
                          'glove-avg-cosine' ]:
            df = pd.DataFrame({'input_id': post_ids, 
                               'model_id': model_name, 
                               'label_id': 'n/a',
                               'misinfo_id': mid, 
                               'confidence': score[0] }).to_dict('records')    
        
        elif model_name in ['bow-log-snli', 'bow-log-mnli','bow-log-mednli', 
                            'boe-log-snli', 'boe-log-mnli','boe-log-mednli', 
                            'bilstm-snli', 'bilstm-mnli', 'bilstm-mednli']:
            
            df = pd.DataFrame({'input_id': post_ids, 
                               'model_id': model_name, 
                               'label_id': preds,
                               'misinfo_id': mid, 
                               'confidence': max_probs, 
                               'misc' : probs.tolist()}).to_dict('records')
    
        elif model_name in ['sbert-snli', 'sbert-mnli', 'sbert-mednli',  
                            'sbert-mednli-ct', 'sbert-snli-ct', 'sbert-mnli-ct', 
                            'sbert-fever-ct', 'sbert-scifact-ct','sbert-ann-ct']:
            df = pd.DataFrame({'input_id': post_ids, 
                               'model_id': model_name, 
                               'label_id': preds.tolist(),
                               'misinfo_id': mid, 
                               'confidence': max_probs.tolist(), 'misc': probs.tolist() }).to_dict('records')
    
        output.append (df)
    
    connection.close()    
    
    #Write to File
    if args.file:
        output_df = pd.DataFrame()
    
        for o in output:
            df = pd.DataFrame.from_records(o)  
            output_df = pd.concat([output_df, df], ignore_index=True)
      
        output_df.to_csv (args.file)
        
        print ('Writing predictions to file is complete...')
    
    #Write to DB
    if args.db_output:
        engine = get_engine(args.db_output)
        connection = engine.connect()
    
        for i in range(len(output)):
            print ('Writing to DB:', i)
            put_outputs(output[i], engine)
    
        connection.close()
        print ('Writing predictions to DB is complete...')


if __name__ == '__main__':
    main()     