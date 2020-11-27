import torch as t
from torch import nn

class LSTMNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim):
        """
        vocab_size: 詞典長度
        embedding_dim: 詞向量的維度
        hidden_dim: LSTM神經元的個數
        layer_dim: LSTM的層數
        output_dim: 輸出的維度
        """
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM + FC
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embeds = self.embedding(x)
        r_out, (h_n, h_c) = self.lstm(embeds, None)  # 全0初始化h0
        # r_out : [batch, time_step, hidden_size]
        # h_n: [n_layers, batch, hidden_size]
        # h_c: [n_layers, batch, hidden_size]
        out = self.fc1(r_out[:, -1, :])  # 選取最後一個時間點的out
        return out