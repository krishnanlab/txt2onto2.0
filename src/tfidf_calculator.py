from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from collections import Counter


class TfidfCalculator:
    '''tf idf tfidf calculator for single words in corpus'''

    def __init__(self, data):
        self.data = data
        self.idf = None
        self.tf = None
        self.tfidf = None
        self.word_features = self.get_word_features()

    def calculate_tfidf(self):
        '''Calculate TF-IDF values for single words in corpus'''
        vectorizer = TfidfVectorizer()
        self.tfidf = vectorizer.fit_transform(self.data).toarray()
        return self.tfidf

    def calculate_tf(self):
        '''Calculate TF values (count) for single words in corpus'''
        vectorizer = CountVectorizer()
        self.tf = vectorizer.fit_transform(self.data).toarray()
        return self.tf

    def calculate_idf(self):
        '''Calculate IDF values for single words in corpus'''
        vectorizer = TfidfVectorizer()
        vectorizer.fit(self.data)
        self.idf = vectorizer.idf_
        return self.idf

    def get_word_features(self):
        '''get word features of returning tf/idf/tfidf matrix'''
        vectorizer = TfidfVectorizer()
        vectorizer.fit(self.data)
        return vectorizer.get_feature_names_out()


class WordClusterTfidfCalculator:
    '''tf idf tfidf calculator for word clusters in corpus'''

    def __init__(self, data, cluster):
        self.data = data
        self.cluster = cluster
        self.idf = None
        self.tf = None
        self.tfidf = None
        self.word_features = self.get_word_features()

    @staticmethod
    def custom_tokenizer(text):
        '''custom tokenizer that can take non-utf-8 words into account'''
        return text.split()

    def calculate_tfidf(self):
        '''
        Calculate TF-IDF values for word clusters
        which is calculated as:
        norm(tf*idf)
        currently only support l2 norm
        '''
        self.calculate_tf()
        self.calculate_idf()
        self.tfidf = self.tf * np.tile(self.idf, (len(self.data), 1))
        self.tfidf = self.tfidf / \
            np.sqrt(np.sum(self.tfidf**2, axis=1))[:, None]

        return self.tfidf

    def calculate_tf(self):
        '''
        Calculate TF values (count) for word clusters
        which is necessary caluclated as the total occurence times of words in
        a sentence
        '''
        vectorizer = CountVectorizer(tokenizer=WordClusterTfidfCalculator.custom_tokenizer)
        tf_ = vectorizer.fit_transform(self.data).toarray()

        self.tf = np.zeros((len(self.data), len(
            np.unique(self.cluster['clusters']))))
        for i, cluster in enumerate(np.unique(self.cluster['clusters'])):
            words_in_cluster = np.array(
                self.cluster[self.cluster['clusters'] == cluster].index)
            words_in_cluster_inds = np.array([np.argwhere(
                self.word_features == word).reshape(-1)[0] for word in words_in_cluster])
            self.tf[:, i] = np.sum(tf_[:, words_in_cluster_inds], axis=1)

        return self.tf

    def calculate_word_tf(self):
        '''Calculate TF values (count) for word'''
        vectorizer = CountVectorizer(tokenizer=WordClusterTfidfCalculator.custom_tokenizer)
        self.tf = vectorizer.fit_transform(self.data).toarray()

        return self.tf

    def calculate_idf(self):
        '''
        Calculate IDF values for word clusters
        which is calculated as:
        idf = log((1+n)/(1+doc frequency of cluster)+1
        '''

        self.idf = np.sum(self.tf > 0, axis=0)
        N = len(self.data)
        self.idf = np.log((N + 1)/(self.idf + 1)) + 1

        return self.idf

    def get_word_features(self):
        '''get word features of returning matrix'''
        vectorizer = TfidfVectorizer(tokenizer=WordClusterTfidfCalculator.custom_tokenizer)
        vectorizer.fit(self.data)

        return vectorizer.get_feature_names_out()


class NER_TfidfCalculator:
    def __init__(self, data):
        '''Calculate TF, IDF, and TFIDF for phrases or words and phrases
        '''
        self.data = data
        self.idf = None
        self.tf = None
        self.tfidf = None
        self.word_features = self.get_word_features()

    def calculate_tf(self):
        self.tf = np.zeros((len(self.data), len(self.word_features)))
        word_to_index = {word: index for index, word in enumerate(self.word_features)}
        for row, line in enumerate(self.data):
            word_count = Counter(line)
            for word, count in word_count.items():
                col = word_to_index[word]
                self.tf[row, col] = count
        return self.tf

    def calculate_idf(self):
        self.idf = np.sum(self.tf > 0, axis=0)
        N = len(self.data)
        self.idf = np.log((N + 1)/(self.idf + 1)) + 1
        return self.idf

    def calculate_tfidf(self):
        self.calculate_tf()
        self.calculate_idf()
        self.tfidf = np.multiply(self.tf, self.idf)
        self.tfidf = self.tfidf/np.sqrt(np.sum(self.tfidf**2, axis=1))[:, None]
        return self.tfidf

    def get_word_features(self):
        self.word_features = []
        for line in self.data:
            self.word_features.extend(line)
        self.word_features = np.array(sorted(list(set(self.word_features))))
        return self.word_features


class PhraseClusterTfidfCalculator:
    '''tf idf tfidf calculator for phrase clusters in corpus'''

    def __init__(self, data, cluster):
        self.data = data
        self.cluster = cluster
        self.idf = None
        self.tf = None
        self.tfidf = None
        self.word_features = self.get_word_features()

    def calculate_tfidf(self):
        '''
        Calculate TF-IDF values for word clusters
        which is calculated as:
        norm(tf*idf)
        currently only support l2 norm
        '''
        self.calculate_tf()
        self.calculate_idf()
        self.tfidf = self.tf * np.tile(self.idf, (len(self.data), 1))
        self.tfidf = self.tfidf / \
            np.sqrt(np.sum(self.tfidf**2, axis=1))[:, None]

        return self.tfidf

    def calculate_phrase_tf(self):
        self.tf = np.zeros((len(self.data), len(self.word_features)))
        word_to_index = {word: index for index, word in enumerate(self.word_features)}
        for row, line in enumerate(self.data):
            word_count = Counter(line)
            for word, count in word_count.items():
                col = word_to_index[word]
                self.tf[row, col] = count
        return self.tf

    def calculate_tf(self):
        '''
        Calculate TF values (count) for word clusters
        which is necessary caluclated as the total occurence times of words in
        a sentence
        '''
        tf_ = self.calculate_phrase_tf()

        self.tf = np.zeros((len(self.data), len(
            np.unique(self.cluster['clusters']))))
        for i, cluster in enumerate(np.unique(self.cluster['clusters'])):
            words_in_cluster = np.array(
                self.cluster[self.cluster['clusters'] == cluster].index)
            words_in_cluster_inds = np.array([np.argwhere(
                self.word_features == word).reshape(-1)[0] for word in words_in_cluster])
            self.tf[:, i] = np.sum(tf_[:, words_in_cluster_inds], axis=1)

        return self.tf

    def calculate_idf(self):
        '''
        Calculate IDF values for word clusters
        which is calculated as:
        idf = log((1+n)/(1+doc frequency of cluster)+1
        '''

        self.idf = np.sum(self.tf > 0, axis=0)
        N = len(self.data)
        self.idf = np.log((N + 1)/(self.idf + 1)) + 1

        return self.idf

    def get_word_features(self):
        self.word_features = []
        for line in self.data:
            self.word_features.extend(line)
        self.word_features = np.array(sorted(list(set(self.word_features))))
        return self.word_features
