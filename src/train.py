'''
This script split data, then train and evaluate model for each task
'''

import model_builder
import numpy as np
import pandas as pd
import pickle
from time import time
from argparse import ArgumentParser
from tfidf_calculator import TfidfCalculator
from pathlib import Path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-input",
                        help="the name of task, like a ontology, sex, age group",
                        required=True,
                        type=str)
    parser.add_argument("-out",
                        help="the dir to studies in each fold",
                        required=True,
                        type=str)
    parser.add_argument("-c",
                        help="regulation strength for LR methods",
                        type=float,
                        default=1.0)
    parser.add_argument("-l1r",
                        help="l1 ratio",
                        type=float,
                        default=1.0)
    args = parser.parse_args()

    # time the code
    t_all = time()

    # get task name from input
    onto = Path(args.input).resolve().stem.split('__')[0]

    # Load input
    desc_df = pd.read_csv(args.input, header=0, index_col=None, sep='\t')
    label = np.array(desc_df['label'])
    text = np.array(desc_df['text'])

    # calculate tfidf and idf of training data
    tfidf_calculator = TfidfCalculator(text)
    trn_tfidf = tfidf_calculator.calculate_tfidf()
    trn_idf = tfidf_calculator.calculate_idf()
    trn_word_features = tfidf_calculator.word_features

    # train model
    model = model_builder.LogisticRegressionModel(C=args.c, l1r=args.l1r)
    model.fit(trn_tfidf, label)

    # save coef, words feature and idf to a dict
    param = {}
    param['coef'] = model.get_coef()
    param['words'] = trn_word_features
    param['idf'] = trn_idf
    param['model'] = model
    param['prior'] = np.mean(label)
    with open(f'{args.out}/{onto}__model.pkl', 'wb') as f:
        pickle.dump(param, f)

    print('took %.2f min to run in total' % ((time()-t_all)/60))
