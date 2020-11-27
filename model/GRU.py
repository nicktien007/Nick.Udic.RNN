import torch as t
from torch import nn


class GRU(nn.Module):
    """
    input_size = 400  # input的維度
    hidden_dim = 400  # 隱藏層維度
    num_layers = 1    # GRU迭代次數
    label_num = 149   # 總Label數量
    max_len = max_len # 句子最大長度->60
    batch_size = 1    # batch_size
    """

    def __init__(self, input_size, hidden_dim, num_layers, output_dim, batch_size, max_len):
        super(GRU, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_size = input_size
        self.batch_size = batch_size
        self.max_len = max_len

        self.gru = nn.GRU(input_size, hidden_dim, num_layers=1, batch_first=True)  # input_size,  隱藏層維度
        self.fc = nn.Linear(hidden_dim, output_dim)  # 將句向量經過一層liner判斷類別

    def forward(self, x):
        # input shape:B * S -> S * B

        h0 = self.__init__hidden()  # 隱藏層h0

        # gru_input = t.randn(self.batch_size, self.max_len, self.input_size)  # batch_size, 句子最大長度, input的維度
        gru_output, hidden = self.gru(x, h0)

        # print(gru_output.shape, hidden.shape)

        sen_vec_output = gru_output[:, -1, :]  # 只使用最後的输出做為句向量
        # print(sen_vec_output.shape)

        liner_output = self.fc(sen_vec_output)
        # print(liner_output.shape)

        return liner_output

    def __init__hidden(self):  # 工具函數，作用是創建初始的隱藏層h0
        hidden = t.zeros(self.num_layers, self.batch_size, self.hidden_dim)  # (num_layers, batch, hidden_dim)
        return hidden.to(t.device("cpu"))  # 加載cpu




