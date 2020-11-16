from torch import nn


# https://bit.ly/34PAvu5
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,  # 雙向LSTM
            batch_first=False
        )
        # fc=>特徵
        self.fc = nn.Linear(
            in_features=embedding_dim * 2,  # embedding_dim*2, 如果bidirectional=false,它就是embedding_dim
            out_features=output_dim,
            bias=True
        )

        # Dropout 避免overfitting
        self.dropout = nn.Dropout(p=0.5, inplace=False)

    def forward(self, x):
        # batch_size = x.size(1)
        output = self.embeddings(x)
        output, (h_n, c_n) = self.rnn(output)
        output = (output[-1, :, :])
        output = self.fc(output)

        return self.dropout(output)
