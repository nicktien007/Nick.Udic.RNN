import logging_utils
from service import tokenize_service as ts, preprocess_dataset_service as pds, train_service as tt,predict_service as ps
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


if __name__ == '__main__':
    logging_utils.Init_logging()
    # test_tokenize()
    # test_preprocess_dataset()
    # test_train_model()
    test_predict()

