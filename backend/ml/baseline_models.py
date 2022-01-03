import os
from dataclasses import dataclass
import pickle
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from backend.ml.detector import Detector
from transformers import AutoModel, AutoTokenizer
import re
from scipy.sparse import hstack

def load_vectorizer (vec_path):
    file = open(vec_path, 'rb')
    vectorizer = pickle.load(file)
    file.close()
    
    return vectorizer

def glove_word_average(embeddings_dict, sentence):
		"""
		Compute average word vector for a single doc/sentence.
		:param sent: list of sentence tokens
		:return:
			mean: float of averaging word vectors
		"""
		mean = []
		sent = sentence.split()
		for word in sent:
			word = re.sub(r"[.,!?\\-\\'\"]", "", word).lower()
			if word.replace('[.!?\\-]', '') in embeddings_dict:
				mean.append(embeddings_dict[word])

		if not mean:  # empty words
			# If a text is empty, return a vector of zeros.
			print("cannot compute average owing to no vector for {}".format(sent))
			return np.zeros(300)
		else:
			mean = np.array(mean).mean(axis=0)
			return mean

###########################################
############# Similarity ##################
###########################################
@dataclass
class BoWCosine(Detector):
    """
    TFIDFDetector
    """    
    def __init__(self, corpus) -> None:        
        Detector.__init__(self)
        self.tfidf_vectorizer = TfidfVectorizer ()
        self.tfidf_vectorizer.fit(corpus)
        
    def _encode(self, sentences):
        vectors = self.tfidf_vectorizer.transform(sentences)
        return vectors

    def _score(self,
               encoded_sentences,
               encoded_misconceptions) -> np.ndarray:
        cos_sim = cosine_similarity (encoded_sentences, encoded_misconceptions)
    
        return cos_sim

class GloVeCosine(Detector):
    """
    TFIDFDetector
    """    
    def __init__(self) -> None:        
        Detector.__init__(self)
        self.embeddings_dict = {}
        with open("models/glove.6B.300d.txt", 'r', encoding='utf8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings_dict[word] = vector
        
    def _encode(self, sentences):
        enc_sentences = []
        for s in sentences:
                enc_sentences.append ( glove_word_average(self.embeddings_dict, s) )
        
        return enc_sentences

    def _score(self,
               encoded_sentences,
               encoded_misconceptions) -> np.ndarray:
        cos_sim = cosine_similarity (encoded_sentences, encoded_misconceptions)
    
        return cos_sim


class BERTCosine(Detector):
    """
    TFIDFDetector

    """    
    def __init__(self, embed_type) -> None:        
        Detector.__init__(self)
        if embed_type == 'base':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')
            self.model = AutoModel.from_pretrained('bert-large-uncased')
        elif embed_type == 'domain_adapted':
            self.tokenizer = AutoTokenizer.from_pretrained('models/roberta-ckpt/covid-roberta')
            self.model = AutoModel.from_pretrained('models/roberta-ckpt/covid-roberta')
        elif embed_type == 'ct-bert':
            self.tokenizer = AutoTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert')
            self.model = AutoModel.from_pretrained('digitalepidemiologylab/covid-twitter-bert')
        
    def _encode(self, sentences):
        results = []
        with torch.no_grad():
            for s in sentences :
                model_input = self.tokenizer.batch_encode_plus([s], pad_to_max_length=True, return_attention_mask=True, return_tensors='pt' )
                model_input = {k: v.to('cuda') for k, v in model_input.items()}
                embeddings, *_ = self.model(**model_input)
                torch.cuda.empty_cache()
                avg_embeddings = embeddings.mean(axis=1).tolist()[0]
                results.append (avg_embeddings)
            return results
    
    #def _encode(self, sentences):
     #   with torch.no_grad():
      #          model_input = self.tokenizer.batch_encode_plus(sentences, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt' )
       #         #model_input = {k: v.to('cuda') for k, v in model_input.items()}
        #        embeddings, *_ = self.model(**model_input)
         #       torch.cuda.empty_cache()
          #      avg_embeddings = embeddings.mean(axis=1).tolist()
        #return avg_embeddings 
    
    def _score(self,
               encoded_sentences,
               encoded_misconceptions) -> np.ndarray:
        cos_sim = cosine_similarity (encoded_sentences, encoded_misconceptions)
    
        return cos_sim

#####################################
########## Entailment ###############
#####################################
class BoWLogistic(Detector):
    """
    TFIDFDetector

    """    
    def __init__(self, model_dir) -> None:        
        Detector.__init__(self)
        path = os.path.join('/',model_dir, 'bow_log_pvect.pkl')        
        self.prem_vectorizer = pickle.load(open(path, "rb"))
        path = os.path.join('/',model_dir, 'bow_log_mvect.pkl')
        self.hyp_vectorizer = pickle.load(open(path, "rb"))
        path = os.path.join('/',model_dir, 'bow_log.pkl')
        self.logreg = pickle.load(open(path, "rb"))
        
    def _encode(self, sentences, sent_type):
        if sent_type == 'prem':
            return self.prem_vectorizer.transform(sentences)
        elif sent_type == 'hyp':
            return self.hyp_vectorizer.transform(sentences)
        
    def _predict(self,
               encoded_premise,
               encoded_hypothesis) -> np.ndarray:
        x_tfidf = hstack([encoded_premise, encoded_hypothesis])
        
        preds = self.logreg.predict(x_tfidf )
        probs = self.logreg.predict_proba(x_tfidf )
                           
        return preds, probs
    
class BoELogistic(Detector):
    """
    TFIDFDetector

    """    
    def __init__(self, model_dir) -> None:        
        Detector.__init__(self)
        self.embeddings_dict = {}
        with open("models/glove.6B.300d.txt", 'r', encoding='utf8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings_dict[word] = vector
        
        path = os.path.join('/',model_dir, 'boe_log.pkl') 
        self.logreg = pickle.load(open(path, "rb"))
        
    def _encode(self, sentences):
        enc_sentences = []
        for s in sentences:
                enc_sentences.append ( glove_word_average(self.embeddings_dict, s) )
        return enc_sentences
        
    def _predict(self,
               encoded_premise,
               encoded_hypothesis) -> np.ndarray:
        
        x = [np.append(i,j) for i, j in zip(encoded_premise, encoded_hypothesis)] 
        x = np.asarray(x, dtype=np.float32)
   
        preds = self.logreg.predict(x)
        probs = self.logreg.predict_proba(x )
             
    
        return preds, probs
 

