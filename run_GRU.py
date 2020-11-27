import gzip

import torch
import torch.nn as nn
import torch.optim as optim
import csv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.nn.utils.rnn import pack_padded_sequence
import math

# 超參數
from service.preprocess_dataset_service import get_taipei_qa_new_train_and_test

HIDDEN_SIZE = 400  # 隱藏層
BATCH_SIZE = 1
N_LAYER = 1  # RNN的層數
N_EPOCHS = 10  # train的輪數
N_CHARS = 400  # 這個就是要構造的字典的長度
USE_GPU = False


# 超參數
# HIDDEN_SIZE = 100 # 隱藏層
# BATCH_SIZE = 256
# N_LAYER = 2 # RNN的層數
# N_EPOCHS = 100 # train的輪數
# N_CHARS = 128 # 這個就是要構造的字典的長度
# USE_GPU = False

# 1：數據集
class NameDataset(Dataset):  # 這個是自己寫的數據集的類，就那3個函數
    def __init__(self, is_train_set=True):
        # x_train, x_test, y_train, y_test = get_taipei_qa_new_train_and_test()
        # filename = "./dataset/Taipei_QA_new_no_lbl.txt"
        filename = "./dataset/Taipei_QA_new_no_lbl.txt"
        with open(filename) as file:
            data = file.readlines()
            x = []
            y = []
            for i, line in enumerate(data):
                sp = line.strip().split(' ')
                x.append(sp[1])  # Q
                y.append(sp[0])  # A
        self.names = x
        self.countries = y

        # filename = "./dataset/names_train.csv.gz" if is_train_set else "./dataset/names_test.csv.gz"
        # with gzip.open(filename, "rt") as f:
        #     reader = csv.reader(f)
        #     rows = list(reader)
        #
        # self.names = [row[0] for row in rows]
        # self.countries = [row[1] for row in rows]

        self.len = len(self.names)
        self.country_list = list(sorted(set(self.countries)))  # 去重+排序
        # self.country_list = y_train  # 去重+排序
        self.country_dict = self.getCountryDict()  # 做一個國家詞典,這個就是標籤 y
        self.country_num = len(self.country_list)

    def __getitem__(self, index):
        return self.names[index], self.country_dict[self.countries[index]]  # 前者是名字字符串，後者是國家的索引

    def __len__(self):
        return self.len

    def getCountryDict(self):
        country_dict = dict()
        for idx, country_name in enumerate(self.country_list, 0):
            country_dict[country_name] = idx
        return country_dict

    def idx2country(self, index):  # 這個就是為了得到分類之後，返回下標對應的字符串，也就是顯示使用的
        return self.country_list[index]

    def getCountriesNum(self):  # 分類的國家數量
        return self.country_num


def make_tensors(names, countries):  # 這個就是將名字的字符串轉換成數字表示
    sequences_and_lengths = [name2list(name) for name in names]  # [(),(),,...]
    name_sequences = [sl[0] for sl in sequences_and_lengths]  # 取轉換成ACCIIS的序列,長度是BatchSize
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths])  # 取序列的長度，轉換成longtensor
    # names.long()
    countries = countries.long()  # 這個cluntries之前轉換成了數字，這裡只轉換成longtensor

    # make tensor of name, BatchSize x SeqLen
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()  # 先做全0的張量，然後填充,長度是BatchSize
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # sort by length to use pack_padded_sequence
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    return create_tensor(seq_tensor), \
           create_tensor(seq_lengths), \
           create_tensor(countries)


def name2list(name):  # 將name字符串的字母轉換成ASCII
    arr = [ord(c) for c in name]
    return arr, len(arr)  # 返回的是元組


def create_tensor(tensor):  # 是否使用GPU
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor


trainset = NameDataset(is_train_set=True)  # train數據
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = NameDataset(is_train_set=False)  # test數據
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

N_COUNTRY = trainset.getCountriesNum()  # 這個就是總的類別的數量


# 2：構造模型
class RNNClassifier(nn.Module):
    """
    這裡的bidirectional就是GRU是不是雙向的，雙向的意思就是既考慮過去的影響，也考慮未來的影響（如一個句子）
    具體而言：正向hf_n=w[hf_{n-1}, x_n]^T,反向hb_0,最後的h_n=[hb_0, hf_n],方括號裡的逗號表示concat。
    """

    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1  # 雙向2、單向1

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,  # 輸入維度、輸出維度、層數、bidirectional用來說明是單向還是雙向
                          bidirectional=bidirectional,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size * self.n_directions, output_size)

    def __init__hidden(self, batch_size):  # 工具函數，作用是創建初始的隱藏層h0
        hidden = torch.zeros(self.n_layers * self.n_directions,
                             batch_size, self.hidden_size)
        return create_tensor(hidden)  # 加載GPU

    def forward(self, input, seq_lengths):
        # input shape:B * S -> S * B
        input = input.t()
        batch_size = input.size(1)

        hidden = self.__init__hidden(batch_size)  # 隱藏層h0
        embedding = self.embedding(input)

        # pack them up
        gru_input = pack_padded_sequence(embedding, seq_lengths)  # 填充了可能有很多的0，所以為了提速，將每個序列以及序列的長度給出

        output, hidden = self.gru(gru_input, hidden)  # 只需要hidden
        if self.n_directions == 2:  # 雙向的，則需要拼接起來
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]  # 單向的，則不用處理
        fc_output = self.fc(hidden_cat)  # 最後來個全連接層,確保層想要的維度（類別數）
        return fc_output


# 4：訓練和測試模型
def trainModel():
    total_loss = 0
    for i, (names, countries) in enumerate(trainloader, 1):  # 記載的下標從1開始
        inputs, seq_lengths, target = make_tensors(names, countries)
        output = classifier(inputs, seq_lengths)  # 預測輸出
        loss = criterion(output, target)  # 求出損失
        optimizer.zero_grad()  # 清除之前的梯度
        loss.backward()  # 梯度反傳
        optimizer.step()  # 更新參數

        total_loss += loss.item()
        if i % 10 == 0:
            print(f'[{time_since(start)}] Epoch {epoch}', end='')
            print(f'[{i * len(inputs)}/{len(trainset)}]', end='')
            print(f'loss={total_loss / (i * len(inputs))}')

    return total_loss


def testModel():
    correct = 0
    total = len(testset)
    print("evaluating trained model...")
    with torch.no_grad():
        for i, (names, countries) in enumerate(testloader, 1):
            inputs, seq_lengths, target = make_tensors(names, countries)  # 將名字的字符串轉換成數字表示
            output = classifier(inputs, seq_lengths)  # 預測輸出
            pred = output.max(dim=1, keepdim=True)[1]  # 預測出來是個向量，裡面的值相當於概率，取最大的
            correct += pred.eq(target.view_as(pred)).sum().item()  # 預測和實際標籤相同則正確率加1

        percent = '%.2f' % (100 * correct / total)
        print(f'Test set:Accuracy{correct} / {total} {percent}%')

    return correct / total


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


if __name__ == "__main__":
    # classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)  # 定義模型
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER, bidirectional=False)  # 定義模型
    if USE_GPU:
        device = torch.device("cuda:0")
        classifier.to(device)

    # 3：定義損失函數和優化器
    criterion = torch.nn.CrossEntropyLoss()  # 分類問題使用交叉熵損失函數
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)  # 使用了隨機梯度下降法

    start = time.time()
    print("Training for %d epochs..." % N_EPOCHS)
    acc_list = []
    for epoch in range(1, N_EPOCHS + 1):
        # Train cycle
        trainModel()
        acc = testModel()
        acc_list.append(acc)  # 存入列表，後面畫圖使用

    # 畫圖
    epoch = np.arange(1, len(acc_list) + 1, 1)  # 步長為1
    acc_list = np.array(acc_list)
    plt.plot(epoch, acc_list)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()  # 顯示網格線 1=True=默認顯示；0=False=不顯示
    plt.show()
