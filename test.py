import tokenize_service as ts
import preprocess_dataset_service as pds


def test_tokenize():
    print(ts.tokenize('傅达仁今将运行安乐死，却突然爆出自己20年前遭纬来体育台封杀，他不懂自己哪里得罪到电视台。'))


def test_preprocess_dataset():
    pds.preprocess_dataset("./dataset/online_shopping_10_cats.csv", './testoutput')


if __name__ == '__main__':
    # test_tokenize()
    test_preprocess_dataset()
