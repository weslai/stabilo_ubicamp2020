import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Encoder basically not change
### Output should be [B, T, H]
class Encoder(nn.Module):
    '''encodes the input'''
    def __init__(self, feature_len, embedding_dim, dropout, batch_size, num_layers=1):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding_dim, self.hidden_size = embedding_dim, 2*embedding_dim
        self.batch_size = batch_size
        self.feature_len = feature_len
        # self.seq_len = seq_len
        self.rnn1 = nn.LSTM(input_size= feature_len, hidden_size= self.hidden_size,
                           num_layers= num_layers, dropout=dropout, batch_first=True)
        self.rnn2 = nn.LSTM(input_size= self.hidden_size, hidden_size= self.embedding_dim,
                            num_layers= num_layers, dropout=dropout, batch_first= True)

    def forward(self, x):
        '''
        Args:
            x: input in batch [batch, time, features]
            lengths: the length of input in batch
            x should be sorted by length by decreasing
        Returns:
            output, final
        '''
        # packed = nn.utils.rnn.pack_padded_sequence(x, seq_len, batch_first=True)
        output, _ = self.rnn1(x)
        output, hidden = self.rnn2(output) ## output_size (packed_batch_size, embedding_dim)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # print('hidden shape')
        # print(hidden.shape())
        return output, hidden

## the simplest Sequence to Sequence Decoder
class Decoder(nn.Module):
    def __init__(self, num_classes, embedded_dim, batch_size, num_layers, dropout, target_size):
        super(Decoder, self).__init__()
        self.feature_len = num_classes
        self.embedded_dim = embedded_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.target_size = target_size
        self.num_classes = num_classes

        self.hidden_size = self.embedded_dim * 2 ## temporary
        self.rnn = nn.LSTM(input_size=self.embedded_dim,
                           hidden_size= self.hidden_size,
                           num_layers=self.num_layers + 1,
                           dropout=self.dropout,
                           bidirectional= False,
                           batch_first=True)

        self.classification = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x): ## take the output from encoder as input
        # x = x.transpose(1, 0)
        x, _ = self.rnn(x) ## x return [Batch, seq_len, hidden_size]
        # x, hidden = self.rnn2(x) ## x return [Batch, seq_len, hidden_size]
        outputs = self.classification(x) ## outputs return [Batch, seq_len, num_classes]
        return outputs

class RecurrentAutoencoder(nn.Module):
    def __init__(self, feature_len, embedding_dim, num_classes, dropout, batch_size, num_layers=1, target_size=6):
        super(RecurrentAutoencoder, self).__init__()
        self.feature_len = feature_len
        self.embedding_dim = embedding_dim
        self.num_classses = num_classes
        self.dropout = dropout
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.target_size = target_size

        self.encoder = Encoder(feature_len=feature_len, embedding_dim=embedding_dim, dropout=dropout,
                               batch_size=batch_size, num_layers=num_layers)
        self.decoder = Decoder(num_classes=num_classes, embedded_dim=embedding_dim, batch_size=batch_size,
                               num_layers=num_layers, dropout=dropout, target_size=target_size)

    def forward(self, x):
        output, x = self.encoder(x)
        x = self.decoder(output)  ##take output gate
        return torch.nn.functional.log_softmax(x, dim=2)  ## (batch_size, output_len, num_classes)