import argparse
from subprocess import call
import os
import torch

from modules.page_retriever_cc import WikiPageRetriever

from modules.embedding_generator_cc import EmbeddingGenerator


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate embeddings of CC Wiki entries.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use. Defaults to `gpu` if one is available, else `cpu`.")
    parser.add_argument("--encoder_model", type=str, default="bert-base-uncased",
                        help="Name of the directory under './model' containing the model to be loaded. If the directory does not exist, the script will attempt to find a mtach on the HuggingFace catalog and download it.")
    parser.add_argument("--tf", type=bool, default=False,
                        help="Set to True to download a model from TensorflowHub instead of HuggingFace")
    parser.add_argument("--target_pages", type=str, default="./target_pages.txt",
                        help="Path to list of target URLs to be acquired and embedded")
    parser.add_argument("--verbose", type=bool, default=True,
                        help="Control output verbosity.")
    return parser.parse_args()


def embed(args):
    # Load Encoder
    if args.verbose:
        print("Initializing encoder.")
    encoder_model = args.encoder_model
    if os.path.isdir('model/' + args.encoder_model):
        encoder_model = 'model/' + encoder_model
    

    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    page_retriever = WikiPageRetriever()
    embedding_generator = EmbeddingGenerator(encoder_model,
                                           device, tf=args.tf)

    # Download pages & run Encoder.
    if args.verbose:
        print("Retrieving Wiki pages.")
    pages = page_retriever(args.target_pages)

    if args.verbose:
        print("Generating Embeddings")
    results = embedding_generator(pages)


def main():
    args = get_args()
    results = embed(args)

    print("Done!")

if __name__ == "__main__":
    main()
