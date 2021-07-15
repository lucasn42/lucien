import streamlit as st
import requests
from PIL import Image
import spacy
#import contextualSpellCheck
import pickle as pkl


def spell_check(query,response):

    for flagged_token in response['flaggedTokens']:

        query = query.replace(flagged_token['token'],flagged_token['suggestions'][0]['suggestion'])

    return query


@st.cache(allow_output_mutation=True)
def load_model():
    nlp = spacy.load("en_core_web_md")
#
#    nlp.add_pipe("contextual spellchecker")
#    
    return nlp

# Stuff we need to use Azure's Spell checker
#TODO: replace with local spell-check model

key = # Azure Cloud key goes here. TODO: read this from an env variable or something else more secure

params = {
    'mkt':'en-us',
    'count': 1,
    'offset': 0,
    }

headers = {
    'Ocp-Apim-Subscription-Key': key,
    }

# Quick CSS hack to hide Streamlit's top bar
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# Here we build the site's sidebar
st.sidebar.markdown("### WHO IS LUCIEN?")

st.sidebar.markdown("[He is a librarian](https://sandman.fandom.com/wiki/Lucien) who lives in a magical castle in the dream world. He's good at finding things.")

st.sidebar.markdown("### How Do I Ask Him Stuff?")
st.sidebar.markdown("**For now, he only takes queries in English. You can formulate a query as a question:**")
st.sidebar.markdown("*\"How can I make my CUDA RNN code behave deterministically?\"*")
st.sidebar.markdown(" ")
st.sidebar.markdown("**As a statement:**")
st.sidebar.markdown("*\"Using multiple GPUs with Pytorch\"*")
st.sidebar.markdown(" ")
st.sidebar.markdown("**Or in keyword form, for example:**")
st.sidebar.markdown("*\"multiple GPUs Tensorflow\"*")
st.sidebar.markdown("### NOTE: Only a few pages from the AI/Machine Learning guide have been included in this demo!")

# Here we build the site's main section
st.title("Ask Lucien")
st.subheader("A Semantic Search Thingy for Compute Canada's Wiki")

input_header = "Type your query here:"

input_value = "Is wandb available on the clusters?"

url = "http://0.0.0.0:5000/search/" # TODO: refactor to read server address from config file.

query = st.text_input(input_header, value=input_value)


if st.button("Search"):

    with st.spinner("Processing input..."):
        
        nlp = load_model()
        doc = nlp(query.lower())  

    state = "good"

#This next commented block is about Spell Checking.
#Top one is a local Spell Checker pipeline based on SpaCy - it's not good.
#Bottom one uses Azure's Bing Search service to leverage their spell checking - it's good, but not free.

    if doc: 
     
        #if doc._.outcome_spellCheck:
        #    st.write("Please try again. Perhaps you mean " + '"' + doc._.outcome_spellCheck + '"' + "?")
        #    state = "bad"

        print("Processing...")

    if True: #else:
            
        #params['q'] = query
        #response = requests.get("https://api.bing.microsoft.com/v7.0/search", headers=headers, params=params).json()


        #if 'alteredQuery' in response['queryContext']:

            #query = spell_check(query.lower(),response)

            #query = response['queryContext']['alteredQuery']

            #st.write(f'Did you mean "{query}" ?')

            #state = "bad"
        print("Processing...")


# Send query to model server, get results back, show 'em to user

    if state=="good": 

        with st.spinner("Scanning Compute Canada wiki. This might take a few minutes..."):

            payload = {'query' : query}

            response = requests.post(url,data=payload).json()

            result_string = "Done!"

            st.markdown(result_string)

            st.subheader(f"Here is what Lucien has found about this subject:" )

            used_titles = []

            all_responses = [article for model in response for article in response[model]]

            sorted_model = sorted(all_responses, key=lambda x: int(x['pertinence_score']), reverse=True)

            st.write(sorted_model)

            ##TODO: List results in a Googl-like way.
            ##This will require work on the html parser and the sentencizer.

            #i=0
                
            #for outputs in sorted_model:

            #    if outputs["title"] not in used_titles:

            #        i += 1

            #        md = f'**{i}:** [{outputs["title"]}]({outputs["url"]})'
            #        md2 = "<ul>" + " ".join([f"<li>{sentence}</li>" for sentence in outputs['sentences']]) + "</ul>"

            #        st.markdown(md)
            #        st.markdown(md2, unsafe_allow_html = True)
            #        st.markdown("---")

            #        used_titles.append(outputs["title"])








