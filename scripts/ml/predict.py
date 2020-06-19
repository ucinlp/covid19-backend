import argparse
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from collections import OrderedDict
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.ml.sentence_bert import SentenceBertClassifier
from backend.ml.misconception import MisconceptionDataset
from backend.stream.db.operation import get_misinfo, get_inputs, put_outputs
from backend.stream.db.util import get_engine
from datetime import datetime


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
    state_dict = torch.load(weights, map_location=torch.device('cpu') )

    # remove `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    
    return model


####################################
########### CMD OPTIONS ############
####################################
parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', type=str, required=True)
parser.add_argument('--weights',type=Path, required=True)
parser.add_argument('--db', type=str, required=True)      
args = parser.parse_args()

#####################################
########### LOAD MODEL ##############
#####################################
model = load_model(args.model_dir, args.weights)
model.eval()

####################################
### Load Misinformation from DB  ###
####################################
misinfos = MisconceptionDataset.from_db(args.db)

mis =[]
mid =[]
for misinfo in misinfos:
    mis.append(misinfo.pos_variations[0] )
    mid.append(misinfo.id)

n = len(mid)

########################################
############ Predict ###################
########################################
engine = get_engine(args.db)
connection = engine.connect()
inputs = get_inputs(engine, connection)

output = []

sm = torch.nn.Softmax(dim=-1)

for input in inputs:
    posts = [input['text']] * n
    post_ids = [input['id']] * n
    with torch.no_grad():
            logits = model(posts, mis)
            _, preds = logits.max(dim=-1)                        
            probs = sm(logits)
            max_probs, _ = probs.max(dim=-1)

    df = pd.DataFrame({'input_id': post_ids, 'model_id': 'sentence-bert', 'label_id': preds.tolist(),
                             'misinfo_id': mid, 'confidence': max_probs.tolist(), 'misc': logits.tolist() }).to_dict('records')
    output.append (df) 

connection.close()

###########################################
####### Write Predictions to DB ###########
###########################################  
connection = engine.connect()

for i in range(len(output)):
    put_outputs(output[i], engine)

connection.close()

print ('Writing predictions to DB is complete...')















