import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from time import time
from tqdm import tqdm
import warnings

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


def ignore_future_warnings(func):
    warnings.simplefilter("ignore", category=FutureWarning)
    return func


@ignore_future_warnings
def generate_embeddings(model_name, text, batch_size=5000):
    '''
    Function to generate embeddings for given text
    '''
    # set device
    def get_device():
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        return device

    device = get_device()
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # load model
    model = AutoModel.from_pretrained(model_name).to(device)

    all_embeddings = []

    if device.type != 'cpu':
        # generate embedding in batch by gpu
        print('generate embedding by gpu')

        # divide text to batches
        batches = [text[i:i + batch_size] for i in range(0, len(text), batch_size)]
        print('batch size is %s, divide text into %s batches' % (batch_size, len(batches)))

        # tokenize input
        print(f'tokenize input for {len(text)} words')

        for idx, batch in enumerate(batches):
            t0 = time()
            encoded_input = tokenizer.batch_encode_plus(batch, padding=True, return_tensors='pt').to(device)
            print(f'tokenize for {(time()-t0)/60} min')
            print(f'generate embedding for {len(batch)} words in batch {idx}')
            t0 = time()
            with torch.inference_mode():
                outputs = model(**encoded_input)
            print(f'generate embedding for batch {idx} in {(time()-t0)/60} min')

            # get cls embedding
            embeddings = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            all_embeddings.append(embeddings)
    else:
        # generate embedding in using cpu
        print('generate embedding by cpu')

        for word in tqdm(text,
                         total=len(text),
                         desc='generate embeddings using cpu'):
            encoded_input = tokenizer(word, return_tensors='pt').to(device)

            with torch.inference_mode():
                outputs = model(**encoded_input)

            embeddings = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()[None, :]
            all_embeddings.append(embeddings)

    return np.concatenate(all_embeddings, axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-input",
                        help="Path to word corpus",
                        required=True,
                        type=str)
    parser.add_argument("-out",
                        help="Output directory to send embedding matrices to",
                        default=None,
                        type=str)
    parser.add_argument("-model",
                        help="Model used to generate embedding, either 'biomedbert_abs', 'biomedbert_full', 'bert'",
                        default='biomedbert_abs',
                        choices=['biomedbert_abs', 'biomedbert_full', 'bert'],
                        type=str)
    parser.add_argument("-batch_size",
                        help="batch size of text when using gpu inference embddings",
                        default=2000,
                        type=int)
    parser.add_argument("-ner",
                        help="if input entitiy is phrase",
                        action='store_true')

    args = parser.parse_args()

    # get name of outfile
    if not args.out:
        outfile = '%s/%s_embeddings.npz' % (Path(args.input).resolve().parent,
                                            Path(args.input).resolve().stem)
    else:
        outfile = args.out

    # get unique words in corpus
    print('get unique words')
    if not args.ner:
        words = unique_words(args.input)
    else:
        words = unique_phrases(args.input)
    print('there are %s unique words in the given descriptions' % (len(words)))

    # load language model
    if args.model.lower() == 'biomedbert_abs':
        model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    elif args.model.lower() == 'biomedbert_full':
        model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    elif args.model.lower() == 'bert':
        model_name = "bert-large-uncased"
    else:
        raise ValueError("Invalid model name. Currently 'biomedbert_abs', 'biomedbert_full', and 'bert' are supported.")

    embeddings_array = generate_embeddings(model_name, words, batch_size=args.batch_size)

    # save output
    print('saving output...')
    np.savez_compressed(outfile,
                        embedding=embeddings_array,
                        words=words)
