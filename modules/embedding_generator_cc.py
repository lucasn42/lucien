import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import List
from tqdm import tqdm
import os
import pickle as pkl
import tensorflow_hub as hub
import tensorflow_text as text

def bert_pipeline(model, preprocessor, claim):

    bert_inputs = preprocessor(claim)

    bert_outputs = model(bert_inputs, training=False)

    output = bert_outputs['pooled_output']
    
    del bert_outputs

    return  output

class EmbeddingGenerator:
    def __init__(self, model_name: str,
                 device: torch.device, tf: bool):
        self.device = device
        self.model = hub.load("https://tfhub.dev/google/" + model_name) if tf else SentenceTransformer(model_name)
        self.bert_pipeline = False
        if "experts/bert" in model_name:
            self.bert_pipeline = True
            self.preprocessor = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
        self.model_path = model_name
        if 'model/' not in model_name:
            self.model_path = 'model/' + model_name
        self.tf = tf

    def __call__(self, documents: List[dict]):
        print("Generating embeddings.")
        i=0
        for document in tqdm(documents):
             print(document['url'])
             if self.tf:
                 if self.bert_pipeline:
                     model_input = self.preprocessor(document['page_one_piece'])
                     sent_embeddings = torch.from_numpy(self.model(model_input,training=False)['pooled_output'].numpy())
                 else:
                     sent_embeddings = torch.from_numpy(self.model(document['page_sentencized']).numpy())
                     doc_embeddings = torch.from_numpy(self.model([document['page_one_piece']]).numpy())

             else:
                with torch.no_grad():
                   sent_embeddings = self.model.encode(document['page_sentencized'],batch_size=128,show_progress_bar=True,convert_to_numpy=False)
                   doc_embeddings = self.model.encode(document['page_one_piece'],batch_size=128,show_progress_bar=True,convert_to_numpy=False)

           #  document = document.copy()
             document['sent_embeddings'] = sent_embeddings
             document['doc_embeddings'] = doc_embeddings

  
             if not os.path.isdir('./embeddings_cc/' + self.model_path):
                  os.makedirs('./embeddings_cc/' + self.model_path)

             file_name = './embeddings_cc/'+ self.model_path + '/document' + str(i) + '.pkl'
             with open(file_name,'wb') as f:
                   pkl.dump(document,f)
                   f.close()
             del sent_embeddings
             del document
             i+=1
                
        return 0
