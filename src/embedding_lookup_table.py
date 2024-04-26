import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from time import time


def unique_words(corpus: str) -> list:
    words = []
    with open(corpus) as f:
        for line in f:
            words.extend(line.rstrip().split(' '))
    words = list(set(words))
    return words


def unique_phrases(corpus: str) -> list:
    words = []
    with open(corpus) as f:
        for line in f:
            words.extend(line.rstrip().split('||'))
    words = list(set(words))
    return words


def generate_embeddings_with_hg_batch(model_name, text):
    '''
    Function to generate embeddings for given text
    '''
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # divide text to batches
    batch_size = 5000
    batches = [text[i:i + batch_size] for i in range(0, len(text), batch_size)]

    # tokenize input
    print(f'tokenize input for {len(text)} words')
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # load model
    model = AutoModel.from_pretrained(model_name).to(device)

    all_embeddings = []
    for batch in batches:
        t0 = time()
        encoded_input = tokenizer.batch_encode_plus(batch, padding=True, return_tensors='pt').to(device)
        print(f'tokenize for {(time()-t0)/60} min')
        # inference
        print(f'generate embedding for {len(batch)} words')
        t0 = time()
        with torch.inference_mode():
            outputs = model(**encoded_input)
        print(f'generate embedding for {(time()-t0)/60} min')

        # get cls embedding
        embeddings = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        all_embeddings.append(embeddings)
    return np.concatenate(all_embeddings, axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-corpus",
                        help="Path to word corpus",
                        required=True,
                        type=str)
    parser.add_argument("-outfile",
                        help="Output directory to send embedding matrices to",
                        default=None,
                        type=str)
    parser.add_argument("-model",
                        help="Model used to generate embedding, either 'pubmedbert_abs', 'pubmedbert_full', 'bert'",
                        default='pubmedbert_abs',
                        type=str)
    parser.add_argument("-ner",
                        help="if input entitiy is phrase",
                        action='store_true')

    args = parser.parse_args()

    # get name of outfile
    if not args.outfile:
        outfile = '%s/%s_embeddings.npz' % (Path(args.corpus).resolve().parent,
                                            Path(args.corpus).resolve().stem)
    else:
        outfile = args.outfile

    # get unique words in corpus
    print('get unique words')
    if not args.ner:
        words = unique_words(args.corpus)
    else:
        words = unique_phrases(args.corpus)

    # load language model
    if args.model.lower() == 'pubmedbert_abs':
        model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    elif args.model.lower() == 'pubmedbert_full':
        model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    elif args.model.lower() == 'bert':
        model_name = "bert-large-uncased"
    else:
        raise ValueError("Invalid model name. Currently 'pubmedbert_abs', 'pubmedbert_full', 'bert' are supported.")

    embeddings_array = generate_embeddings_with_hg_batch(model_name, words)

    # save output
    print('saving output...')
    np.savez_compressed(outfile,
                        embedding=embeddings_array,
                        words=words)
