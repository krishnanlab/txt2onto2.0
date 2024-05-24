import pickle
import numpy as np
import pandas as pd
from time import time
from scipy.sparse import csr_matrix
from pathlib import Path
from argparse import ArgumentParser
from tfidf_calculator import TfidfCalculator


def word_to_inds(words, word_features):
    '''Function to convert words to indices based on a word feature list'''
    idx_dict = dict(zip(words, np.arange(len(words))))
    return np.array([idx_dict[i] for i in word_features])


def sparse_mat_mul(a, b):
    '''Function to perform sparse matrix multiplication'''
    sparse_matrix_a = csr_matrix(a)
    sparse_matrix_b = csr_matrix(b)
    result_sparse = sparse_matrix_a.dot(sparse_matrix_b)
    return result_sparse.toarray()


def calc_cosine_similarity(emb1, emb2):
    '''Function to calculate cosine similarity between two embeddings'''
    return np.dot(emb1, emb2.T) / np.dot(
        np.linalg.norm(emb1, axis=1)[:, None],
        np.linalg.norm(emb2, axis=1)[None, :])


def load_embedding(file):
    '''Function to load embedding from a file'''
    file_ = np.load(file)
    return file_['embedding'], file_['words']


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-input",
                        help="input text for prediction",
                        required=True,
                        type=str)
    parser.add_argument("-out",
                        help="outdir to prediction results",
                        required=True,
                        type=str)
    parser.add_argument("-id",
                        help="input instance id",
                        required=True,
                        type=str)
    parser.add_argument("-input_embed",
                        help="word embedding for the input text",
                        required=True,
                        type=str)
    parser.add_argument("-train_embed",
                        help="word embedding for training text",
                        required=True,
                        type=str)
    parser.add_argument("-model",
                        help="trained model",
                        required=True,
                        type=str)
    args = parser.parse_args()

    ts = time()
    print('loading data')
    t0 = time()

    # get task name from input
    onto = Path(args.model).resolve().stem.split('__')[0]

    # load samples
    samples = []
    with open(args.id) as f:
        for line in f:
            samples.append(line.rstrip().split('\t')[0])
    if len(samples) != len(set(samples)):
        raise Exception(f'IDs provided in {args.id} is not unique\n')
    samples = np.array(samples)

    # load model
    with open(args.model, 'rb') as f:
        param = pickle.load(f)
    coef = param['coef']
    model_features = param['words']
    idf = param['idf']
    model = param['model']
    prior = param['prior']

    # load embeddings lookup table for input text
    input_embed, input_words = load_embedding(args.input_embed)

    # load embeddings lookup table for training text
    train_embed, train_words = load_embedding(args.train_embed)

    # load input for prediction
    desc_df = pd.read_csv(args.input, header=None, index_col=None, sep='\t')

    # runtime
    print('took %.2f s to load data' % ((time()-t0)))

    print('predicting labels')
    t0 = time()

    # calculate TF for input text
    tfidf_calculator = TfidfCalculator(np.array(desc_df[0]))
    tf = tfidf_calculator.calculate_tf()
    input_features = tfidf_calculator.get_word_features()

    # load embeddings for tf matrix of input text
    input_embed = input_embed[word_to_inds(input_words, input_features), :]

    # load embedding of words in features of classification model
    train_embed = train_embed[word_to_inds(train_words, model_features), :]

    # get cloest words
    cosine_similarity = calc_cosine_similarity(input_embed, train_embed)
    max_values = np.max(cosine_similarity, axis=1)
    max_ind_matrix = np.where(cosine_similarity == max_values.reshape(-1, 1), 1, 0)

    # get tfidf matrix of input desc
    input_tfidf = sparse_mat_mul(tf, max_ind_matrix) * np.tile(idf, (tf.shape[0], 1))
    input_tfidf = input_tfidf / np.sqrt(np.sum(input_tfidf**2, axis=1))[:, None]

    # predict for all samples
    preds = model.predict(input_tfidf)

    # runtime
    print('took %.2f s to predict' % ((time()-t0)))

    print('retrieving predictive words')
    t0 = time()
    # get and save related words
    related_word_idx = np.argwhere(coef > 0).reshape(-1)
    related_words = sparse_mat_mul(tf, max_ind_matrix[:, related_word_idx])
    related_words_idxs = np.argwhere(related_words > 0)
    max_ind_matrix_filtered = max_ind_matrix[:, related_word_idx].T

    related_word_dict = {}
    for i, j in zip(related_words_idxs[:, 0], related_words_idxs[:, 1]):
        related_words_idx = np.argwhere((tf[i] * max_ind_matrix_filtered[j]) > 0).reshape(-1)
        related_word_dict.setdefault(samples[i], [])
        related_word_dict[samples[i]].extend(input_features[related_words_idx])

    # runtime
    print('took %.2f s to retrieve predictive words' % ((time()-t0)))

    print('saving output')
    # save prediction
    df = pd.DataFrame(zip(samples, preds, np.log2(preds/prior))).sort_values(by=1, ascending=False)
    df[3] = [','.join(related_word_dict[i]) if i in related_word_dict else '' for i in df[0]]
    df.columns = ['ID', 'prob', 'log2(prob/prior)', 'related_words']
    df.to_csv(
        '%s/%s__preds.csv' % (args.out, onto),
        header=True,
        index=False,
    )

    # runtime
    print('took %.2f min to load, predict, retrieve predictive words and save %s for %s instances' % ((time()-ts)/60, onto, len(samples)))
