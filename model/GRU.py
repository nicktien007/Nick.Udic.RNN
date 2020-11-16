import torch as t
from torch import nn


class GRU(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers,  label_num, max_len):
        super(GRU, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_size = input_size
        self.max_len = max_len

        # self.embedding = nn.Embedding(input_size, hidden_dim)
        self.gru = nn.GRU(input_size, hidden_dim, num_layers=1, batch_first=True)  # input_size,  隱藏層維度
        self.fc = nn.Linear(hidden_dim, label_num) # 將句向量經過一層liner判斷類別

        # h0 = t.zeros(num_layers, batch_size, hidden_dim)  # (num_layers, batch, hidden_dim)

    def forward(self, x):
        # input shape:B * S -> S * B
        x = x.t()
        batch_size = x.size(1)

        h0 = self.__init__hidden(batch_size)  # 隱藏層h0
        # embedding = self.embedding(input)

        gru_input = t.randn(batch_size, self.max_len, self.input_size)  # batch_size, 句子最大長度, input的維度

        gru_output, hidden = self.gru(gru_input, h0)

        print(gru_output.shape, hidden.shape)

        sen_vec_output = gru_output[:, -1, :]  # 只使用最後的输出做為句向量

        print(sen_vec_output.shape)

        liner_output = self.fc(sen_vec_output)
        print(liner_output.shape)

        return liner_output

    def __init__hidden(self, batch_size):  # 工具函數，作用是創建初始的隱藏層h0
        hidden = t.zeros(self.num_layers, batch_size, self.hidden_dim)
        return hidden.to(t.device("cpu"))  # 加載cpu
