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

    t0 = time()

    # get task name from input
    onto = Path(args.model).resolve().stem.split('__')[0]

    # load samples
    samples = np.loadtxt(args.id, delimiter='\t', dtype=str)

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

    # get and save related words
    # get index of predictive word features
    related_word_idx = np.argwhere(coef > 0).reshape(-1)
    # get mapping of words in the given text to predictive words in models
    max_ind_matrix_filtered = max_ind_matrix[:, related_word_idx].T
    related_word_dict = {}
    for idx, ind_matrix in enumerate(max_ind_matrix_filtered):
        # For each predictive words in the corpus, find related words in external data
        related_words = tf * np.tile(ind_matrix[None, :], (tf.shape[0], 1))
        # get index of samples with related words
        related_sample_idxs = np.argwhere(np.sum(related_words, axis=1) > 0).reshape(-1)
        # get sample and words
        for related_sample_idx in related_sample_idxs:
            related_word_dict.setdefault(samples[related_sample_idx], [])
            related_word_dict[samples[related_sample_idx]].extend(list(input_features[related_words[related_sample_idx] > 0]))

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
    print('took %.2f min to predict %s for %s instances' % ((time()-t0)/60, onto, len(samples)))
