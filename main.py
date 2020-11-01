import torch
import logging_utils

from service import train_service
from service.tokenize_service import tokenize
from service.train_service import build_RNN_model,setup_manual_seed

device = torch.device('cpu')

def main():
    # preprocess_dataset("./dataset/online_shopping_10_cats.csv", "./train_data")
    setup_manual_seed()
    build_RNN_model('./train_data', './trained_model',device)
    predict()


def predict_sentiment(net, vocab, sentence):
    train_service.eval()
    # 分詞
    tokenized = tokenize(sentence)
    # sentence 的索引
    indexed = [vocab.stoi[t] for t in tokenized]

    tensor = torch.LongTensor(indexed).to(device)  # seq_len
    tensor = tensor.unsqueeze(1)  # seq_len * batch_size (1)

    # tensor寫text一樣的tensor
    prediction = torch.sigmoid(net(tensor))

    return prediction.item()





def predict():
    net = torch.load('trained_model/RNN-model.pt')
    vocab = torch.load('./trained_model/vocab')

    score = predict_sentiment(net, vocab, '最差的一次购物、快递都用了九天、什么心情都没有了')

    print(score, "正面" if score > 0.5 else "負面")


if __name__ == '__main__':
    logging_utils.Init_logging()
    main()
