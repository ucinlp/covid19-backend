import os
import pickle
import json
import argparse
import re
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

def map_values(row, values_dict):
    return values_dict[row]

values_dict = {'neutral': 2, 
               'entailment': 0, 
               'contradiction': 1, 
               '-' : 3}

def read_json (file):
    data = []
    with open(file) as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame.from_records(data)[['sentence1', 'sentence2', 'gold_label']]
    df['gold_label'] = df['gold_label'].map(values_dict)  
    df = df[df['gold_label'].isin([0, 1, 2])]
    return df

def sent_word_avg_embed(embeddings_dict, sentence):                                                                                                   
    mean = []
    sent = sentence.split()
    for word in sent:
        word = re.sub(r"[.,!?\\-\\'\"]", "", word).lower()
        if word.replace('[.!?\\-]', '') in embeddings_dict:
            mean.append(embeddings_dict[word])

    if not mean:  # empty words
		# If a text is empty, return a vector of zeros.
        return np.zeros(300)
    else:
        mean = np.array(mean).mean(axis=0)
        return mean
  
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--dev', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--c', type=int, default=1)
    parser.add_argument('--feature-type', type=str, required=True) #bow or boe

    args = parser.parse_args()  
    
    train_df = read_json(args.train)
    dev_df = read_json(args.dev)    
    
    if args.feature_type=='bow':
        kwargs = {
        'ngram_range' : (1,2),
        'dtype' : 'int32',
        'strip_accents' : 'unicode',
        'decode_error' : 'replace',
        'analyzer' : 'word',
        'min_df' : 2
        }
    
        # Vectorizers for Sentence1    
        p_vectorizer =  TfidfVectorizer(**kwargs)
        px_train = p_vectorizer.fit_transform(train_df['sentence1'])
        px_dev = p_vectorizer.transform(dev_df['sentence1'])    
        filename = os.path.join('/',args.output_dir, 'pvect.pkl')
        pickle.dump(p_vectorizer, open(filename, "wb"))
        
        # Vectorizers for Sentence2    
        m_vectorizer =  TfidfVectorizer(**kwargs)
        mx_train = m_vectorizer.fit_transform(train_df['sentence2'])
        mx_dev = m_vectorizer.transform(dev_df['sentence2'])    
        filename = os.path.join('/',args.output_dir, 'mvect.pkl')
        pickle.dump(m_vectorizer, open(filename, "wb"))
        
        x_train = hstack([px_train, mx_train])
        x_dev = hstack([px_dev, mx_dev])
        
        model_filename = os.path.join('/',args.output_dir, 'bow_log.pkl')
        
    elif args.feature_type=='boe':
        embeddings_dict = {}
        with open("models/glove.6B.300d.txt", 'r', encoding='utf8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector
                
        #Avg. Embeddings for Sentence1    
        px_train = []
        px_dev = []
        for sentence in train_df.sentence1:
            px_train.append (sent_word_avg_embed(embeddings_dict, sentence))            
        for sentence in dev_df.sentence1:
            px_dev.append (sent_word_avg_embed(embeddings_dict, sentence))
        # Avg. Embeddings for Sentence2
        mx_train = []
        mx_dev = []        
        for sentence in train_df.sentence2:
          mx_train.append (sent_word_avg_embed(embeddings_dict, sentence))    
        for sentence in dev_df.sentence2:
          mx_dev.append (sent_word_avg_embed(embeddings_dict, sentence))
        
        x_train = [np.append(i,j) for i, j in zip(px_train, mx_train)]
        x_train = np.asarray(x_train, dtype=np.float32)
        x_dev = [np.append(i,j) for i, j in zip(px_dev, mx_dev)]        
        x_dev = np.asarray(x_dev, dtype=np.float32)
        
        model_filename = os.path.join('/',args.output_dir, 'boe_log.pkl')
    
    print("Training ...")
    logreg = LogisticRegression(C=args.c, penalty='l2', solver = 'saga', max_iter = 100)
    logreg.fit(x_train, train_df['gold_label'])

    train_accuracy = logreg.score(x_train, train_df['gold_label'])
    dev_accuracy = logreg.score(x_dev, dev_df['gold_label'])

    print("Training Accuracy = ", train_accuracy,
      "\nDev Accuracy = ", dev_accuracy)

    pickle.dump(logreg, open(model_filename, 'wb'))
    
if __name__ == '__main__':
    main()