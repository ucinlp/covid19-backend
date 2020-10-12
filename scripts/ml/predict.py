# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 20:14:00 2020

@author: taman
"""

from backend.stream.db.table import  Output
from backend.stream.db.util import get_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import select
from sqlalchemy import and_
import torch
from backend.ml.bertscore import BertScoreDetector
import pandas as pd
from backend.stream.db.operation import get_inputs, put_outputs
from backend.ml.misconception import MisconceptionDataset
from backend.ml.sentence_bert import SentenceBertClassifier
import argparse
from backend.ml.baseline_models import BoWCosine, GloVeCosine, BERTCosine, BoWLogistic, BoELogistic
from pathlib import Path
from backend.ml.bi_lstm import BiLSTM
from pyserini.search import SimpleSearcher
####################################
########### CMD OPTIONS ############
####################################
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--db_input', type=str, required=True)
parser.add_argument('--file', type=Path, required=False)
parser.add_argument('--db_output', type=str,  required=False)

args = parser.parse_args()
model_name = args.model_name
db_input = args.db_input

###################
#### Functions ####
###################
       
###################
#### Functions ####
###################
       
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
  
    model = torch.load(model_path)
    model.field = torch.load (tokenizer_path, pickle_module=dill)
    return model
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

############################
##### Load Models ##########
############################
if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

################
## Similarity ##
################
if model_name == 'glove-avg-cosine':
    model = GloVeCosine()

elif model_name == 'bert-base-cosine':
    model = BERTCosine('base')
    model.model.eval()
    model.model.cuda()

elif model_name == 'bert-ft-cosine':
    model = BERTCosine('domain_adapted')
    model.model.eval()
    model.model.cuda()
    
elif model_name == 'ct-bert-cosine':
    model = BERTCosine('ct-bert')
    model.model.eval()
    model.model.cuda()
    
elif model_name == 'bert-score-base':
  model = BertScoreDetector('bert-large-uncased')
  model.eval()
  #model.cuda()
elif model_name == 'bert-score-ft':
  model = BertScoreDetector('models/roberta-ckpt/covid-roberta')
  model.eval()
  #model.cuda()
elif model_name == 'bert-score-ct':
  model = BertScoreDetector('digitalepidemiologylab/covid-twitter-bert')
  model.eval()
  #model.cuda()
################
## Entailment ##
################
# BoW Logistic
elif model_name == 'bow-log-snli':  
    model = BoWLogistic('snli')
    
elif model_name == 'bow-log-mnli':  
    model = BoWLogistic('mnli')
    
elif model_name == 'bow-log-mednli':  
    model = BoWLogistic('mednli')

# BoE Logistic
elif model_name == 'boe-log-snli':  
    model = BoELogistic('snli')
    
elif model_name == 'boe-log-mnli':  
    model = BoELogistic('mnli')
    
elif model_name == 'boe-log-mednli':  
    model = BoELogistic('mednli')

elif model_name == 'sbert-mnli':    
    model = load_sbert_model('bert-base-cased', 'models/mnli-sbert-ckpt-1.pt')
    model.cuda()
    model.eval()
    sm = torch.nn.Softmax(dim=-1)
    
elif model_name == 'sbert-snli':    
    model = load_sbert_model('bert-base-cased', 'models/snli-sbert-ckpt-2.pt')
    model.cuda()
    model.eval()
    sm = torch.nn.Softmax(dim=-1)

elif model_name == 'sbert-mednli':   
  
    model = load_sbert_model('bert-base-cased', 'models/SBERT-MEDNLI-ckpt-5.pt')
    model.cuda()
    model.eval()
    sm = torch.nn.Softmax(dim=-1)

elif model_name == 'sbert-snli-ct':   
  
    model = load_sbert_model('digitalepidemiologylab/covid-twitter-bert', 'models/SBERT-SNLI-ckpt-2.pt')

    model.cuda()
    model.eval()
    sm = torch.nn.Softmax(dim=-1)

elif model_name == 'sbert-mnli-ct':   
  
    model = load_sbert_model('digitalepidemiologylab/covid-twitter-bert', 'models/SBERT-MNLI-ckpt-2.pt')
    model.cuda()
    model.eval()
    sm = torch.nn.Softmax(dim=-1)

elif model_name == 'sbert-mednli-ct':   
  
    model = load_sbert_model('digitalepidemiologylab/covid-twitter-bert', 'models/SBERT-ct-mednli-ckpt-7.pt')

    model.cuda()
    model.eval()
    sm = torch.nn.Softmax(dim=-1)

    
elif model_name == 'bilstm-snli':    
    model = load_bilstm ('models/bilstm-snli.pt', 'models/bilstm-snli-field.pt' )
    model.cuda()
    model.eval()
    sm = torch.nn.Softmax(dim=-1)

elif model_name == 'bilstm-mnli':    
    model = load_bilstm ('models/bilstm-mnli.pt', 'models/bilstm-mnli-field.pt' )
    model.cuda()
    model.eval()
    sm = torch.nn.Softmax(dim=-1)

elif model_name == 'bilstm-mednli':    
    model = load_bilstm ('models/bilstm-mednli.pt', 'models/bilstm-mednli-field.pt' )
    model.cuda()
    model.eval()
    sm = torch.nn.Softmax(dim=-1)

elif model_name == 'bilstm-fnc':    
    model = load_bilstm ('models/fnc/bilistm-fnc.pt', 'models/fnc/bilstm-fnc-field.pt' )
    model.cuda()
    model.eval()
    sm = torch.nn.Softmax(dim=-1)


#############################
#### Read Misinformation ####
#############################
misinfos = MisconceptionDataset.from_db(db_input)

mis = []
mid = []

for misinfo in misinfos:
    mis.append(misinfo.pos_variations[0] )
    mid.append(misinfo.id)

n = len(mid)

# Encode misconception
if model_name in ['bert-base-cosine', 'bert-ft-cosine', 'ct-bert-cosine', 'bert-score-base', 'bert-score-ft',
                  'bert-score-ct','glove-avg-cosine', 'boe-log-snli', 'boe-log-mnli', 'boe-log-mednli' ,
                  'bilstm-snli', 'bilstm-mnli', 'bilstm-mednli']:
    mis_vect = model._encode (mis)

elif model_name in ['bow-log-snli', 'bow-log-mnli', 'bow-log-mednli']:
    mx = model._encode (mis, 'hyp')
      
#########################
###### Predict ##########
#########################
# Tweets
engine = get_engine(db_input)
connection = engine.connect()
inputs = get_inputs(engine, connection)

#output = pd.DataFrame()
output = []

for input in inputs:
    print (input['id'] )
    posts = [input['text']] * n
    post_ids = [input['id']] * n
    
    ###########################################################
    ################ SIMILARITY MODELS ########################
    ###########################################################

    #############################
    #### Bag of Words Cosine ####
    #############################
    if model_name == 'bow-cosine':
      corpus = mis + [input['text']]
      model = BoWCosine(corpus)
      mis_vect = model._encode(mis)      

    ###############################
    ######## Output ###############
    ###############################
    
    if model_name in ['bow-cosine', 'bert-base-cosine', 'bert-ft-cosine', 'ct-bert-cosine', 'bert-score-base', 'bert-score-ft','bert-score-ct','glove-avg-cosine' ]:
        post_vect = model._encode([input['text']])
        score = model._score(post_vect, mis_vect)
        df = pd.DataFrame({'input_id': post_ids, 'model_id': model_name, 'label_id': 'n/a',
                             'misinfo_id': mid, 'confidence': score[0] }).to_dict('records')
    
    ###########################################################
    ################ ENTAILMENT MODELS ########################
    ###########################################################
    
    ###########################################
    ######### BOW - Logistic ##################
    ###########################################
    if model_name in ['bow-log-snli', 'bow-log-mnli', 'bow-log-mednli']:
        px = model._encode(posts, 'prem')
     
        preds, probs = model._predict(px,mx)    
        max_probs = probs.max(axis =1)            
       
    ################################################
    ############ BOE - Logistic ####################
    ################################################
    elif model_name in ['boe-log-snli', 'boe-log-mnli', 'boe-log-mednli']:
        post_vect = model._encode([input['text']]) * n
        
        preds, probs = model._predict(post_vect, mis_vect)    
        max_probs = probs.max(axis =1) 
    
    ################################################
    ################## BiLSTM ######################
    ################################################
    elif model_name in ['bilstm-snli', 'bilstm-mnli', 'bilstm-mednli', 'bilstm-fnc']:
        post_vect = model._encode([input['text']]* n) 
        with torch.no_grad():
            logits = model(post_vect, mis_vect)  
            probs = sm(logits)
            max = probs.max(dim=-1)
            max_probs = max[0].tolist()
            preds = max[1].tolist()
              
    ################################################
    ############## Sentence BERT ###################
    ################################################
    elif model_name in ['sbert-snli', 'sbert-mnli', 'sbert-mednli', 'sbert-mednli-ct', 'sbert-mnli-ct', 'sbert-snli-ct']:
        with torch.no_grad():
            logits = model(posts, mis)           
            _, preds = logits.max(dim=-1)                        
            probs = sm(logits)
            max_probs, _ = probs.max(dim=-1)
    
    ###############################
    ######## Output ###############
    ###############################
    
    if model_name in ['bow-cosine', 'bert-base-cosine', 'bert-ft-cosine', 'ct-bert-cosine', 'bert-score-base', 'bert-score-ft','bert-score-ct', 'glove-avg-cosine' ]:
        df = pd.DataFrame({'input_id': post_ids, 'model_id': model_name, 'label_id': 'n/a',
                             'misinfo_id': mid, 'confidence': score[0] }).to_dict('records')    
    
    elif model_name in ['bow-log-snli', 'bow-log-mnli','bow-log-mednli', 'boe-log-snli', 'boe-log-mnli','boe-log-mednli', 'bilstm-snli', 'bilstm-mnli', 'bilstm-mednli']:
        
        df = pd.DataFrame({'input_id': post_ids, 'model_id': model_name, 'label_id': preds,
                             'misinfo_id': mid, 'confidence': max_probs, 'misc' : probs.tolist()}).to_dict('records')

    elif model_name in ['sbert-snli', 'sbert-mnli', 'sbert-mednli', 'sbert-mednli-ct', 'sbert-snli-ct', 'sbert-mnli-ct']:
        df = pd.DataFrame({'input_id': post_ids, 'model_id': model_name, 'label_id': preds.tolist(),
                             'misinfo_id': mid, 'confidence': max_probs.tolist(), 'misc': probs.tolist() }).to_dict('records')

    output.append (df)

connection.close()


#############################################
####### Write Predictions to File ###########
#############################################
if args.file:
    output_df = pd.DataFrame()

    for o in output:
        df = pd.DataFrame.from_records(o)  
        output_df = pd.concat([output_df, df], ignore_index=True)
  
    output_df.to_csv (args.file)
    
    print ('Writing predictions to file is complete...')

###########################################
####### Write Predictions to DB ###########
###########################################  
if args.db_output:
    engine = get_engine(args.db_output)
    connection = engine.connect()

    for i in range(len(output)):
        print ('Writing to DB:', i)
        put_outputs(output[i], engine)

    connection.close()
    print ('Writing predictions to DB is complete...')


      