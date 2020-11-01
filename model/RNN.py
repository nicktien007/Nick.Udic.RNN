from torch import nn


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
        output = self.embeddings(x)
        output, (h_n, c_n) = self.rnn(output)
        output = (output[-1, :, :])
        output = self.fc(output)

        return self.dropout(output)
