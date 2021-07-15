# Semantic Search on Compute Canada's Wiki

Lucien allows users to ask questions in natural language (English only for now) and get answers straight off of Compute Canada's Wiki!


## Usage

Deploying Lucien takes only three steps:

1- Generate [Embeddings](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture) out of the content of target Wiki pages.

2- Start the model server

3- Start the web application server

### Generating Embeddings

Use the script <code>generate_embeddings_cc.py</code> to automatically download the source HTML of the pages listed in <code>target_pages.txt</code>, parse them into text, and output a pickled list of dictionaries containing some page metadata, the text content of the pages and embeddings of them. You can pass any text encoder available on the [HuggingFace](https://huggingface.co) or [Tensorflow Hub](https://tfhub.dev) catalogs into the script to generate the embeddings. For example:

<code>generate_embeddings_cc.py --encoder_model="universal-sentence-encoder/5" --tf=True --verbose=True --target_pages="target_pages.txt"</code>

Will generate embeddings of the pages listed in <code>target_pages.txt</code> using Google's [Universal Sentence Encoder](https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder)


### Starting the model server

The model server is a [Flask](https://flask.palletsprojects.com/en/2.0.x/) application that exposes the Machine Learning models Lucien uses to find content. You will need a web server to run this application. While Flask has a built-in web server, its use is recommended only for development/testing. You can run it with:

<code>python model_server.py</code> 


### Starting the web application server

The actual user-facing part of this application was built with [Streamlit](https://streamlit.io).You can run it with:

<code>streamlit run ask_lucien.py --server.port=80</code>
