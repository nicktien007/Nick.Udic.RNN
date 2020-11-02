import torch

from service import train_service
from service.tokenize_service import tokenize


def predict_sentiment(net, vocab, sentence):
    net.eval()
    # 分詞
    tokenized = tokenize(sentence)
    # sentence 的索引
    indexed = [vocab.stoi[t] for t in tokenized]

    device = torch.device('cpu')
    tensor = torch.LongTensor(indexed).to(device)  # seq_len
    tensor = tensor.unsqueeze(1)  # seq_len * batch_size (1)

    # tensor寫text一樣的tensor
    prediction = torch.sigmoid(net(tensor))

    return prediction.item()


def predict(input_model_path,sentence):
    net = torch.load(input_model_path+'/RNN-model.pt')
    vocab = torch.load(input_model_path+'/vocab')

    score = predict_sentiment(net, vocab, sentence)

    print(score, "正面" if score > 0.5 else "負面")