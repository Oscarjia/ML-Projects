# -*- coding: utf-8 -*-
"""
# @Time    : 2019/12/2 19:03
# @Author  : zhaobinghao
# @File    : auto_abstract_new.py
"""
import sys
import re
import data_io, params, SIF_embedding
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import data_io, params, SIF_embedding
import numpy as np


class AutoAbstract(object):

    def __init__(self):
        self.words_chi = None
        self.We_chi = None
        self.word2weight_chi = None
        self.words_eng = None
        self.We_eng = None
        self.word2weight_eng = None

    def load_model(self):
        sys.path.append('../src')
        weightpara = 1e-5  # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
        print('读取中文模型')
        self.words_chi, self.We_chi = data_io.getWordmap('../models/wiki_news_model2_vector.txt')
        self.word2weight_chi = data_io.getWordWeight('../models/word_count.txt',  # each line is a word and its frequency,
                                                     weightpara)  # word2weight['str'] is the weight for the word 'str'
        print('中文模型读取完毕')
        print('读取英文模型')
        weightpara = 1e-3
        self.words_eng, self.We_eng = data_io.getWordmap('../models/glove_large.txt')
        self.word2weight_eng = data_io.getWordWeight('../models/enwiki_vocab_min200.txt', # each line is a word and its frequency
                                                     weightpara)  # word2weight['str'] is the weight for the word 'str'
        print('英文模型读取完毕')

    def get_embedding(self, sentences, language='Chinese', weightpara=1e-3):
        """
        This function return the embeddings for all sentences in the input parameter: sentences
        sentences is a list of sentencs need for SIF embeddings
        """
        if language == 'Chinese':
            # word vector file
            # For model2：
            # wordfile =
            # For model1：
            # wordfile='../models/wiki_news_word_vector_small2.txt'
            # word frequency file
            # weightfile =
            words = self.words_chi
            word2weight = self.word2weight_chi
            We = self.We_chi
        else:
            # for english use:
            # wordfile =
            # wordfile='../models/glove.840B.300d.txt'
            # weightfile =
            # weightpara = 1e-5
            # weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
            words = self.words_eng
            word2weight = self.word2weight_eng
            We = self.We_eng
        rmpc = 1  # number of principal components to remove in SIF weighting scheme

        weight4ind = data_io.getWeight(words, word2weight)
        print('weight4ind finished ')

        # load sentences
        if language == 'Chinese':
            x, m = data_io.sentences2idx_c(sentences, words)
        else:
            x, m = data_io.sentences2idx(sentences, words)
        # print (x.shape) # (句子的数量，最长的句子的单词的数量)
        # print (m.shape) # (句子的数量，最长的句子的单词的数量)

        print('sentences2idx finished ')
        w = data_io.seq2weight(x, m, weight4ind)  # get word weights

        print('seq2weight finished ')

        # set parameters
        param = params.params()
        param.rmpc = rmpc

        # get SIF embedding
        """
        return 所有需要计算similarity的title，全文，句子的embedding。
        paper 里面用的是TruncatedSVD，project 要求我们用PCA方法decomposite
        """
        print('embedding start ')
        embedding = SIF_embedding.SIF_embedding(We, x, w, param,
                                                method='PCA')  # embedding[i,:] is the embedding for sentence i

        print('embedding finished ')
        print(embedding.shape)
        return embedding

    # 2: Get similarity scores for splited sentences
    def get_similarity_vector(self, title, content, seperator, weightpara, language='Chinese'):
        """
        This function get similarity scores for each splited sentences with the title and content of news
        """
        print('Title is:', title, '\n')
        if language == 'Chinese':
            clean_content = content.replace(' ', '').replace('\r\n', '')
        else:
            clean_content = content.replace('\r\n', '')
        # clean_content=re.findall(r'([\u4e00-\u9fff]+|[A-Z]+|[0-9]+|[a-z])',content)
        print('Content is :', clean_content, '\n')
        # print ('Seperators are:', seperator,'\n')
        splited_content = list(filter(None, re.split(seperator, clean_content)))
        all_sentences = [title, content] + splited_content
        # print (all_sentences)
        # for a, b  in enumerate(all_sentences):
        # print (a, ' : ', len(b), ' ',  b + '\n')

        print('There are ', str(len(all_sentences) - 2), 'sentences been splited and need embeddings.')
        embedding = self.get_embedding(all_sentences, language, weightpara)
        emb_t = embedding[0]
        emb_c = embedding[1]
        scores = []
        for i, b in enumerate(all_sentences[2:]):
            # print ('sentences',i, ':', b)
            emb_i = embedding[i + 2]

            scores.append(0.5 * (pearsonr(emb_t, emb_i)[0] + pearsonr(emb_c, emb_i)[0]))
        # print ('scores for all the sentences are:',  scores)
        return scores, all_sentences

    # Step3:  Conduct KNN smooth for scores
    @staticmethod
    def knn_smooth(scores, n_neigbors=3):
        adjusted_scores = []
        for i, s in enumerate(scores):
            scores_range = scores[max(0, i - n_neigbors):i + n_neigbors + 1]
            mean = np.sum(scores_range) / len(scores_range)
            if s < mean:
                adj_s = s + np.abs(mean - s) * 0.4
            else:
                adj_s = s - np.abs(mean - s) * 0.4
            adjusted_scores.append(adj_s)
        return adjusted_scores

    # plot original scores Vs Smoonthed scores
    @staticmethod
    def plot_bar_x(scores, adjusted_scores):
        plt.figure(figsize=(8, 8))
        label = list(range(len(scores)))
        # this is for plotting purpose
        index1 = np.arange(len(label)) * 3
        # print (index1)
        index2 = index1 + 1
        # print (index2)
        plt.bar(index1, scores, label='Original Scores')
        plt.bar(index2, adjusted_scores, label='Smoothed Scores')
        plt.xlabel('Sentences', fontsize=15)
        plt.ylabel('Similarity scores', fontsize=15)
        plt.xticks(index1, label, fontsize=15, rotation=30)
        plt.title('Original scores Vs Smoothed scores')
        plt.legend()
        plt.show()

    # Step 4: return certain % of sentences based on the number of splited sentences
    @staticmethod
    def return_sentences(adjusted_scores, pct_keep):
        # print ('smoothed scores:',adjusted_scores)
        num_of_sentences = min(15, max(1, int(len(adjusted_scores) * pct_keep)))
        abs_index = sorted(range(len(adjusted_scores)), key=lambda i: adjusted_scores[i], reverse=True)[
                    :num_of_sentences]
        return abs_index

    def get_abstract(self, title, content, seperator, n_neigbors, pct_keep, weightpara, language):
        """
        title: Title of the news
        content: content of the news
        seperator: seperators that been used to split the content into sentences
        n_neigbors: number of sentences to used for KNN smoothing
        pct_keep: % of sentences to keep for the abstract
        language: the language for the news
        """
        abstract = []
        scores, all_scentences = self.get_similarity_vector(title, content, seperator, weightpara, language)
        adjusted_scores = self.knn_smooth(scores, n_neigbors)
        # self.plot_bar_x(scores, adjusted_scores)
        abs_idx = self.return_sentences(adjusted_scores, pct_keep)
        print('Abstract of this news is :')
        for _, value in enumerate(sorted(abs_idx)):
            # print (value)
            print(all_scentences[value + 2])
            abstract.append(''.join([s.strip() for s in all_scentences[value + 2].split()]))
        print('The abstract contains:', len(abstract), ' splited sentences.')
        return ' '.join(abstract)


if __name__ == '__main__':
    title = '小米MIUI 9首批机型曝光：共计15款'
    content = '此外，自本周（6月12日）起，除小米手机6等15款机型外，其余机型已暂停更新发布（含开发版/体验版内测，稳定版暂不受影' \
              '响），以确保工程师可以集中全部精力进行系统优化工作。有人猜测这也是将精力主要用到MIUI 9的研发之中。MIUI 8去年5月' \
              '发布，距今已有一年有余，也是时候更新换代了。当然，关于MIUI 9的确切信息，我们还是等待官方消息。'
    # title = 'k-nearest neighbors algorithm '
    # content = 'In pattern recognition, the k-nearest neighbors algorithm (k-NN) is a non-parametric method used for classification and regression! In both cases, the input consists of the k closest training examples in the feature space! The output depends on whether k-NN is used for classification or regression: In k-NN classification, the output is a class membership, An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor. The neighbors are taken from a set of objects for which the class (for k-NN classification) or the object property value (for k-NN regression) is known. This can be thought of as the training set for the algorithm, though no explicit training step is required, A peculiarity of the k-NN algorithm is that it is sensitive to the local structure of the data.'
    instance = AutoAbstract()
    instance.load_model()
    # instance.get_abstract(title, content, seperator=r'。|\，|\！|\……|\（|\）|\？|\.|\,|\!|\?|\(|\)', n_neigbors=3,
    #                       pct_keep=0.5,
    #                       weightpara=1e-3, language='eng')
    instance.get_abstract(title, content, seperator=r'。|\，|\！|\……|\（|\）|\？|\.|\,|\!|\?|\(|\)', n_neigbors=5, pct_keep=0.5,
                 weightpara=1e-5, language='Chinese')
