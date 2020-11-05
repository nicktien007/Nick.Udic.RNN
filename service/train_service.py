import logging as log
import os

import numpy as np
import torch
from torchtext import data

from model.RNN import RNN
from service.tokenize_service import tokenize


def build_RNN_model(train_data_path, trained_model_path, device):
    if not os.path.isdir(trained_model_path):
        os.mkdir(trained_model_path)

    setup_manual_seed()

    TEXT, test_data, train_data = get_build_RNN_model_data(train_data_path, trained_model_path)

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

    # 因為我們輸出只有1個，所以選用binary cross entropy
    loss_func = torch.nn.BCEWithLogitsLoss().to(device)

    # 用cpu
    rnn.to(device)
    # 開始訓練
    log.info('start train')
    for epoch in range(10):
        train(rnn, epoch, train_iterator, optimizer, loss_func, trained_model_path)
        eval(rnn, epoch, test_iterator, loss_func)
    log.info('end train')
    torch.save(rnn, trained_model_path + '/RNN-model.pt')  # 存模型


def get_build_RNN_model_data(train_data_path, trained_model_path):
    log.info("start get_build_RNN_model_data")
    CAT = data.Field(sequential=True, fix_length=20)
    LABEL = data.LabelField(dtype=torch.float)
    TEXT = data.Field(tokenize=tokenize,  # 斷詞function
                      lower=True,  # 是否將數據轉小寫
                      fix_length=100,  # 每條數據的長度
                      stop_words=None)
    train_data, valid_data, test_data = \
        data.TabularDataset.splits(
            path=train_data_path + "/",  # 數據所在文件夾
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

    torch.save(TEXT.vocab, trained_model_path + "/vocab")
    log.info("end get_build_RNN_model_data")

    return TEXT, test_data, train_data


def binary_acc(preds, y):
    """
    準確率
    """
    preds = torch.round(torch.sigmoid(preds))
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(rnn, epoch, iterator, optimizer, loss_func, trained_model_path):
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
        torch.save(rnn, trained_model_path + '/RNN-model.pt')  # 存模型
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


def setup_manual_seed():
    # 設置固定生成隨機數的種子，使得每次運行文件時生成的隨機數相同
    if torch.cuda.is_available():
        print("gpu cuda is available!")
        torch.cuda.manual_seed(1000)
    else:
        print("cuda is not available! cpu is available!")
        torch.manual_seed(1000)
