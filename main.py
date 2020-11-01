import pandas as pd
import numpy as np
import torch
from torch import nn
from torchtext import data
import logging_utils
import logging as log

from tokenize_service import tokenize

device = torch.device('cpu')


def main():
    # preprocess_dataset()
    setup_manual_seed()
    # build_RNN_model()
    do_predict()


def get_build_RNN_model_data():
    log.info("start get_build_RNN_model_data")
    CAT = data.Field(sequential=True, fix_length=20)
    LABEL = data.LabelField(dtype=torch.float)
    TEXT = data.Field(tokenize=tokenize,  # 斷詞function
                      lower=True,  # 是否將數據轉小寫
                      fix_length=100,  # 每條數據的長度
                      stop_words=None)
    train_data, valid_data, test_data = \
        data.TabularDataset.splits(
            path='./train_data/',  # 數據所在文件夾
            train='dataset_train.csv',
            validation='dataset_valid.csv',
            test='test.csv',
            format='csv',
            skip_header=True,
            fields=[('cat', CAT), ('label', LABEL),
                    ('review', TEXT)])
    print('len of train_data:', len(train_data))
    print('len of test data:', len(test_data))
    print('len of valid_data:', len(valid_data))
    print(train_data.examples[18].review)
    print(train_data.examples[18].label)
    # 建立詞典
    # TEXT.build_vocab(train_data, vectors="glove.6B.100d")#可以使用預訓練word vector
    TEXT.build_vocab(train_data, max_size=10000)
    LABEL.build_vocab(train_data)
    CAT.build_vocab(train_data)
    print(TEXT.vocab.freqs.most_common(20))  # 資料集裡最常出現的20個詞
    print(TEXT.vocab.itos[:10])  # 列表 index to word
    print(TEXT.vocab.stoi)  # 字典 word to index

    torch.save(TEXT.vocab, "./trained_model/vocab")
    log.info("end get_build_RNN_model_data")

    return TEXT, test_data, train_data


def setup_manual_seed():
    # 設置固定生成隨機數的種子，使得每次運行文件時生成的隨機數相同
    if torch.cuda.is_available():
        print("gpu cuda is available!")
        torch.cuda.manual_seed(1000)
    else:
        print("cuda is not available! cpu is available!")
        torch.manual_seed(1000)


def build_RNN_model():
    TEXT, test_data, train_data = get_build_RNN_model_data()

    batch_size = 30
    # # Iterator是torchtext到模型的輸出
    train_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, test_data),
        batch_size=batch_size,
        sort_key=lambda x: len(x.review),
        device=device
    )
    vocab_size = len(TEXT.vocab)  # 詞典大小
    # vocab_size = 10002  # 詞典大小
    embedding_dim = 100  # 詞向量維度
    hidden_dim = 100  # 隱藏層維度
    output_dim = 1  # 輸出層
    rnn = RNN(vocab_size, embedding_dim, hidden_dim, output_dim)
    print(rnn)
    # pretrained_embedding = TEXT.vocab.vectors
    # print('pretrained_embedding:', pretrained_embedding.shape)
    # rnn.embedding.weight.data.copy_(pretrained_embedding)
    # print('embedding layer inited.')
    # 優化器
    optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-3)
    loss_func = torch.nn.BCEWithLogitsLoss().to(device)  # 因為我們輸出只有1個，所以選用binary cross entropy
    # 用cpu
    rnn.to(device)
    # 開始訓練
    log.info('start train')
    for epoch in range(10):
        train(rnn, epoch, train_iterator, optimizer, loss_func)
        eval(rnn, epoch, test_iterator, loss_func)
    log.info('end train')
    torch.save(rnn, 'trained_model/RNN-model.pt')  # 存模型


class RNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=False
        )
        self.fc = nn.Linear(
            in_features=200,
            out_features=1,
            bias=True
        )

        self.dropout = nn.Dropout(p=0, inplace=False)

    def forward(self, x):
        # batch_size = x.size(1)
        x_ = self.embeddings(x)
        x_, (h_n, c_n) = self.rnn(x_)
        x_ = (x_[-1, :, :])
        x_ = self.fc(x_)
        x_ = self.dropout(x_)

        return x_


def binary_acc(preds, y):
    """
    準確率
    """
    preds = torch.round(torch.sigmoid(preds))
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(rnn, epoch, iterator, optimizer, loss_func):
    rnn.train()  # 訓練模式

    avg_acc = []
    TrainAcc = 0.0
    TrainLoss = 0.0

    for step, batch in enumerate(iterator):
        optimizer.zero_grad()  # 優化器優化之前須將梯度歸零

        # [seq, b] => [b, 1] => [b]
        pred = rnn(batch.review).squeeze(1)

        acc = binary_acc(pred, batch.label).item()  # 取得正確率
        TrainAcc = TrainAcc + acc
        avg_acc.append(acc)

        loss = loss_func(pred, batch.label)  # loss計算
        TrainLoss = TrainLoss + loss

        loss.backward()  # 反向傳遞
        optimizer.step()  # 通過梯度做參數更新(更新權重)

        # if step%20 == 0:print('train loss: %.4f, train acc: %.2f' %(loss, acc))

    TrainLoss = TrainLoss / (step + 1)  # epoch loss
    TrainAcc = TrainAcc / (step + 1)  # epoch acc
    avg_acc = np.array(avg_acc).mean()
    log.info('epoch : %d ,TrainLoss: %f , TrainAcc acc: %f , avg TrainAcc acc: %f ' % (
        epoch + 1, round(TrainLoss.item(), 3), round(TrainAcc, 3), avg_acc))

    if avg_acc > 0.9:
        torch.save(rnn, 'trained_model/RNN-model.pt')  # 存模型
        # torch.save(rnn.state_dict(), SYS_DIR+trained_Model_Path+'/'+trained_Model_Name) #只存權重
        log.info('save done')


def eval(rnn, epoch, iterator, loss_func):
    rnn.eval()  # 評估模式

    avg_acc = []
    TestAcc = 0.0
    TestLoss = 0.0

    with torch.no_grad():
        for step, batch in enumerate(iterator):
            # [b, 1] => [b]
            pred = rnn(batch.review).squeeze(1)

            acc = binary_acc(pred, batch.label).item()  # 取得正確率
            TestAcc = TestAcc + acc
            avg_acc.append(acc)

            loss = loss_func(pred, batch.label)  # loss計算
            TestLoss = TestLoss + loss

            # if step%20 == 0:print('test loss: %.4f, test acc: %.2f' %(loss, acc))

        TestLoss = TestLoss / (step + 1)
        TestAcc = TestAcc / (step + 1)
        avg_acc = np.array(avg_acc).mean()
        log.info('epoch : %d ,TestLoss: %f , TestAcc acc: %f , avg TestAcc acc: %f ' % (
            epoch + 1, round(TestLoss.item(), 3), round(TestAcc, 3), avg_acc))


def predict_sentiment(net, vocab, sentence):
    net.eval()
    # 分詞
    tokenized = tokenize(sentence)
    # sentence 的索引
    indexed = [vocab.stoi[t] for t in tokenized]

    tensor = torch.LongTensor(indexed).to(device)  # seq_len
    tensor = tensor.unsqueeze(1)  # seq_len * batch_size (1)

    # tensor寫text一樣的tensor
    prediction = torch.sigmoid(net(tensor))

    return prediction.item()


# 資料預處理
def preprocess_dataset():
    pd_all = pd.read_csv('./dataset/online_shopping_10_cats.csv')
    print('評論數目（全）：%d' % pd_all.shape[0])
    print('評論數目（正向）：%d' % pd_all[pd_all.label == 1].shape[0])
    print('評論數目（負向）：%d' % pd_all[pd_all.label == 0].shape[0])
    print(type(pd_all))
    # print(pd_all.sample(20))
    # 構造平衡語料
    pd_positive = pd_all[pd_all.label == 1]
    pd_negative = pd_all[pd_all.label == 0]
    pd_60000 = get_balance_corpus(60000, pd_positive, pd_negative)
    # pd_60000 = get_balance_corpus(1000, pd_positive, pd_negative)
    # print(pd_60000.sample(20))
    # 先將數據分為train.csv和test.csv
    split_dataFrame(df=pd_60000,
                    trainfile='./train_data/train.csv',
                    valtestfile='./train_data/test.csv',
                    seed=999,
                    ratio=0.2)
    # 再將train.csv分為dataset_train.csv和dataset_valid.csv
    split_csv(infile='./train_data/train.csv',
              trainfile='./train_data/dataset_train.csv',
              valtestfile='./train_data/dataset_valid.csv',
              seed=999,
              ratio=0.2)
    # print(tokenize('傅达仁今将运行安乐死，却突然爆出自己20年前遭纬来体育台封杀，他不懂自己哪里得罪到电视台。'))


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


def do_predict():
    net = torch.load('trained_model/RNN-model.pt')
    vocab = torch.load('./trained_model/vocab')

    score = predict_sentiment(net, vocab, '最差的一次购物、快递都用了九天、什么心情都没有了')

    print(score, "正面" if score > 0.5 else "負面")


if __name__ == '__main__':
    logging_utils.Init_logging()
    main()
