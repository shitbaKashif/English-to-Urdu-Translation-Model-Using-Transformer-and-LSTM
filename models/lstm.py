import torch
import torch.nn as nn

class LSTMTranslator(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size=256, hidden_size=512, num_layers=2):
        super().__init__()
        self.encoder = nn.Embedding(src_vocab_size, embed_size)
        self.lstm_encoder = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

        self.decoder = nn.Embedding(tgt_vocab_size, embed_size)
        self.lstm_decoder = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, tgt_vocab_size)

    def forward(self, src, tgt):
        src_emb = self.encoder(src)
        _, (hidden, cell) = self.lstm_encoder(src_emb)

        tgt_emb = self.decoder(tgt)
        output, _ = self.lstm_decoder(tgt_emb, (hidden, cell))
        output = self.fc_out(output)
        return output
