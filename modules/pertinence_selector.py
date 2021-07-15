from sentence_transformers import CrossEncoder, util
from typing import List
from tqdm import tqdm
import os
import torch


class PertinenceSelector:
    def __init__(self, model: str, threshold: float):
        #self.tokenizer = AutoTokenizer.from_pretrained(model)
        #self.model = AutoModelForSequenceClassification.from_pretrained(model).eval().to(self.device)
        self.model_name = model

        if os.path.isdir("model/" + model):
            model = "model/" + model
        self.model = CrossEncoder(model)
        
        self.threshold = threshold

    def __call__(self, query: str, corpus: List[dict]):

        print("Computing pertinence.")

        model_inputs = [[query[0]," ".join(match['sentences']) if len(match['sentences'])>1 else match['sentences'][0] if match['sentences'] else match['title']] for match in corpus]

        print(model_inputs)
        
        with torch.no_grad():
              scores = list(map(torch.sigmoid,self.model.predict(model_inputs,convert_to_numpy=False)))


        relevant_idx = [x for x in range(0,len(scores)) if scores[x]>self.threshold]

        print(f"found {len(relevant_idx)} articles")

        for x in relevant_idx:

            corpus[x]['pertinence_score'] = scores[x].item()


        relevant = [corpus[i] for i in relevant_idx]


        return relevant
