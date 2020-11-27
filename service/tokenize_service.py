import codecs
import jieba.posseg as psg

from udicOpenData.dictionary import *
from udicOpenData.stopwords import *

from opencc import OpenCC

from gensim import models

jieba.set_dictionary('./dict/dict.txt.big.txt')
cc = OpenCC('s2t')  # s2t: 簡體中文 -> 繁體中文


# 中文斷詞
def tokenize(sentence):
    stopword_set = get_stopword_set()

    converted = cc.convert(sentence)  # 簡>繁
    words = jieba.posseg.cut(converted)
    words = [w.word for w in words if w.word not in stopword_set]
    return words


def get_stopword_set():
    stopword_set = set()
    with codecs.open('./dict/stopwords.txt', 'r', 'utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))
    return stopword_set


def to_vector(words):
    vector = []
    model = models.Word2Vec.load('./trained_model/wiki_model/word2vec_wiki_zh.model.bin')

    for i, q in enumerate(words):
        word_list = list(rmsw(q))
        tmp = []
        for word in word_list:
            try:
                tmp.append(model[word])
            except:
                continue
        vector.append(tmp)

    return vector
