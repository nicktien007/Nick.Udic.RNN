import os

import numpy as np
import pandas as pd


# 資料預處理
def preprocess_dataset(input_file, output_path, corpus_size):
    pd_all = pd.read_csv(input_file)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    print('評論數目（全）：%d' % pd_all.shape[0])
    print('評論數目（正向）：%d' % pd_all[pd_all.label == 1].shape[0])
    print('評論數目（負向）：%d' % pd_all[pd_all.label == 0].shape[0])
    print(type(pd_all))
    # print(pd_all.sample(20))
    # 構造平衡語料
    pd_positive = pd_all[pd_all.label == 1]
    pd_negative = pd_all[pd_all.label == 0]
    pd_corpus = get_balance_corpus(corpus_size, pd_positive, pd_negative)
    # pd_60000 = get_balance_corpus(1000, pd_positive, pd_negative)
    # print(pd_60000.sample(20))
    # 先將數據分為train.csv和test.csv
    split_dataFrame(df=pd_corpus,
                    trainfile=output_path + '/train.csv',
                    valtestfile=output_path + '/test.csv',
                    seed=999,
                    ratio=0.2)
    # 再將train.csv分為dataset_train.csv和dataset_valid.csv
    split_csv(infile=output_path + '/train.csv',
              trainfile=output_path + '/dataset_train.csv',
              valtestfile=output_path + '/dataset_valid.csv',
              seed=999,
              ratio=0.2)


def get_balance_corpus(corpus_size, corpus_pos, corpus_neg):
    sample_size = corpus_size // 2
    pd_corpus_balance = pd.concat([corpus_pos.sample(sample_size, replace=corpus_pos.shape[0] < sample_size), \
                                   corpus_neg.sample(sample_size, replace=corpus_neg.shape[0] < sample_size)])

    print('評論數目(總體)：%d' % pd_corpus_balance.shape[0])
    print('評論數目(正向)：%d' % pd_corpus_balance[pd_corpus_balance.label == 1].shape[0])
    print('評論數目(負向)：%d' % pd_corpus_balance[pd_corpus_balance.label == 0].shape[0])

    return pd_corpus_balance


def split_csv(infile, trainfile, valtestfile, seed=999, ratio=0.2):
    df = pd.read_csv(infile)
    split_dataFrame(df, trainfile, valtestfile, seed, ratio)


def split_dataFrame(df, trainfile, valtestfile, seed=999, ratio=0.2):
    df["review"] = df.review.str.replace("\n", " ")
    idxs = np.arange(df.shape[0])
    np.random.seed(seed)
    np.random.shuffle(idxs)
    val_size = int(len(idxs) * ratio)
    df.iloc[idxs[:val_size], :].to_csv(valtestfile, index=False)
    df.iloc[idxs[val_size:], :].to_csv(trainfile, index=False)