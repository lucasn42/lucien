import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from typing import List
from tqdm import tqdm
import os
import pickle as pkl

import tensorflow_hub as hub

def bert_pipeline(model, preprocessor, query):

    bert_inputs = preprocessor([query])

    bert_outputs = model(bert_inputs, training=False)
    return  bert_outputs['pooled_output']

class SentenceMatcher:
    def __init__(self, model: str, threshold: float,
                 device: torch.device, tf: bool):
        self.device = device
        #self.tokenizer = AutoTokenizer.from_pretrained(model)
        #self.model = AutoModelForSequenceClassification.from_pretrained(model).eval().to(self.device)
        self.model_name = model

        if tf:
           self.model = hub.load("https://tfhub.dev/google/" + model)
           self.tf = tf
           self.bert_pipeline = False
           if "experts/bert" in model:
               self.bert_pipeline = True
               self.preprocessor = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')

        else:
           if os.path.isdir("model/" + model):
              model = "model/" + model
           self.model = SentenceTransformer(model)
           self.tf = False

        self.threshold = threshold


        documents = []
        for file in os.listdir("embeddings/model/" + self.model_name):
            with open('embeddings/model/' + self.model_name + '/' + file,'rb') as f:
                document = pkl.load(f)
                del document['doc_embeddings']
                del document['page_one_piece']

                documents.append(document)

        self.documents = documents

    def __call__(self, query: str, corpus: List[dict], k=10,score_function=util.cos_sim):
        print("Selecting rationales.")
        results = []
        if self.tf:
           if self.bert_pipeline:
               query_embedding = torch.from_numpy(bert_pipeline(self.model,self.preprocessor,query).numpy())
           else:    
               query_embedding = torch.from_numpy(self.model(query).numpy())
        else:
           with torch.no_grad():
              query_embedding = self.model.encode(query,convert_to_tensor=True)
  
        for article in corpus:
            
            try:

            #if 1==1:
                
                matched_doc = self.documents[article['corpus_id']].copy()
                sent_embeddings = list(filter(lambda tensor : torch.numel(tensor)>0 ,list(matched_doc['sent_embeddings']) ))
                matches = util.semantic_search(query_embedding, sent_embeddings, top_k=k, score_function=score_function)
                matched_doc['sentences'] = [[d['corpus_id'] for d in matches[i] if d['score']>self.threshold] for i in range(0,len(matches))]
                matched_doc['sentences_confidence'] = [[str(d['score']) for d in matches[i] if d['score']>self.threshold] for i in range(0,len(matches))]
                sentences = [[matched_doc['page_sentencized'][j] for j in matched_doc['sentences'][i]] for i in range(0,len(matched_doc['sentences']))]
                matched_doc['sentences'] = sentences[0]
                del matched_doc['sent_embeddings']
                del matched_doc['page_sentencized']
                results.append(matched_doc)

            except:
                print("Skipping Document...")

        return results
