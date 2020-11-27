import logging as log

import logging_utils
from service import tokenize_service as ts, preprocess_dataset_service as pds, train_service as tt, \
    predict_service as ps
import torch


def test_tokenize():
    print(ts.tokenize('傅达仁今将运行安乐死，却突然爆出自己20年前遭纬来体育台封杀，他不懂自己哪里得罪到电视台。'))


def test_preprocess_dataset():
    pds.preprocess_dataset("./dataset/online_shopping_10_cats.csv", './test_output')


def test_train_model():
    device = torch.device('cpu')
    # tt.build_RNN_model('./test_output', './test_train_model', device)
    tt.build_RNN_model('./train_data', './test_train_model', device)


def test_predict():
    ps.predict("./trained_model", "掉色很严重 垃圾货 服务质量差")


def test_embedding():
    embedding = torch.nn.Embedding(5, 3, padding_idx=0)
    inputs = torch.tensor(
        [1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    print(embedding(inputs))


if __name__ == '__main__':
    logging_utils.Init_logging()
    # test_tokenize()
    # test_preprocess_dataset()
    # test_train_model()
    # test_predict()
    # test_get_taipei_qa_new_train_and_test()
    # eval_RNN()
    # test_embedding()
    # test_train_GRU()
    # test_csv()
    # test_step2()
    # preprocess_dataset_qa()

# 寫在colad上面，之後要來驗
# 使用實驗室api斷詞 https://github.com/UDICatNCHU/UdicOpenData
# !pip3 install udicOpenData
# from udicOpenData.dictionary import *
# from udicOpenData.stopwords import *
#
# # device = torch.device('cpu')
# net = torch.load(SYS_DIR+"taipei_qa_GRU.pt").to(device)
# vocab = torch.load(SYS_DIR+'vocab')
#
# input = "●僑民役男：「歸國僑民及原具香港、澳門地區僑民身分之役男，依法尚不須辦理徵兵處理者，如何申辦出境？」"
#
# tokenized = list(rmsw(input))
# indexed = [vocab.stoi[t] for t in tokenized]
# print(indexed)
# tensor = torch.LongTensor(indexed).to(device)  # seq_len
# print(tensor.size())
# tensor = tensor.unsqueeze(1)
# pred = net(tensor)
# print(pred)
# pred = torch.argmax(pred).item()
#
# # pred = torch.argmax(pred).item()
# print(pred)
