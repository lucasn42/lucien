import json
from flask import Flask, render_template, request, jsonify,flash,redirect,session,url_for
import requests
import os
from sentence_transformers import util
import torch
import tensorflow as tf

from modules.document_selector import DocumentSelector
from modules.sentence_matcher_doc import SentenceMatcher
from modules.pertinence_selector import PertinenceSelector

# To prevent Tensorflow from allocating 100% of the GPU memory to a single encoder, in case we want to use multiple encoders
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize encoders and load embeddings in memory
#TODO: write cleaner way to load and use multiple encoders.

document_selector_1 = DocumentSelector("allenai/specter",threshold=0.6)

pertinence_selector = PertinenceSelector('cross-encoder/ms-marco-MiniLM-L-6-v2',threshold=0.985)

#sentence_selector_3 = SentenceMatcher("facebook-dpr-question_encoder-multiset-base",selection_method="topk",device='cpu',tf=False,threshold=74)

#sentence_selector_2 = SentenceMatcher("universal-sentence-encoder/5",device='cpu',tf=True,threshold=0.5)

sentence_selector_1 = SentenceMatcher("msmarco-distilbert-base-tas-b",device='cpu',tf=False,threshold=95)


# Given a query, find good candidate wiki pages, look for candidate answers and evaluate relevance of the answers.

def inference(query, device):

  documents = document_selector_1(query)

  results1 = pertinence_selector(query,sentence_selector_1(query, documents, score_function=util.dot_score))

  #results2 = pertinence_selector(query,sentence_selector_2(query, documents))

  #results3 = pertinence_selector(query,sentence_selector_3(query, documents,score_function=util.dot_score))

  return [results1]#, results2], results3]


# Create API endpoint and expose inference function: 

def create_app():

  app = Flask(__name__)
  app.secret_key = os.urandom(12)

  @app.route('/')
  def index():
     return redirect(url_for('search'))

  @app.route('/search/', methods=['POST'])
  def square():
      sentence = request.form.get('query',0)

      results = inference([sentence],device)

      data = {'model1':results[0]}#,'model2':results[1]}#,'model3':results[2]}
      data = jsonify(data)

      return data

  return app

if __name__ == '__main__':
    app = create_app()
    app.secret_key = os.urandom(12)
    app.run(debug=True,host='0.0.0.0',port=5000, use_reloader=False)
