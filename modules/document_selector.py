import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from typing import List
from tqdm import tqdm
import os
import pickle as pkl


class DocumentSelector:
    def __init__(self, model: str, threshold: float):
        #self.tokenizer = AutoTokenizer.from_pretrained(model)
        #self.model = AutoModelForSequenceClassification.from_pretrained(model).eval().to(self.device)
        self.model_name = model

        if os.path.isdir("model/" + model):
            model = "model/" + model
        self.model = SentenceTransformer(model)

        self.threshold = threshold

        doc_embeddings = []
        for file in os.listdir("embeddings/model/" + self.model_name):
            with open('embeddings/model/' + self.model_name + '/' + file,'rb') as f:
                doc_embeddings.append(pkl.load(f)['doc_embeddings'])

        self.doc_embeddings = doc_embeddings


    def __call__(self, query: str):
        print("Selecting documents.")
        with torch.no_grad():
            query_embedding = self.model.encode(query,convert_to_tensor=True)

            matches = util.semantic_search(query_embedding, self.doc_embeddings, top_k=k)

            matches = [d for d in matches[0] if  d['score'] > self.threshold]

            print(f"Found {len(matches)} documents!")

        return matches
