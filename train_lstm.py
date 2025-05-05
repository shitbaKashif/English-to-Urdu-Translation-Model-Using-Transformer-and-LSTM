import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.tokenizer import BPETokenizer  # Updated import
from utils.dataset import TranslationDataset
from models.lstm import LSTMTranslator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=0)
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=0)
    return src_batch.T, tgt_batch.T

# Load Tokenizers
src_tokenizer = BPETokenizer()
src_tokenizer.load('models/tokenizer_en.pt')  # Load English tokenizer
tgt_tokenizer = BPETokenizer()
tgt_tokenizer.load('models/tokenizer_ur.pt')  # Load Urdu tokenizer

train_dataset = TranslationDataset('data/bible/train.en', 'data/bible/train.ur', src_tokenizer, tgt_tokenizer)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

model = LSTMTranslator(
    src_vocab_size=src_tokenizer.vocab_size,
    tgt_vocab_size=tgt_tokenizer.vocab_size
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

losses = []

for epoch in range(20):
    model.train()
    epoch_loss = 0
    for src, tgt in train_loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()
        output = model(src, tgt_input)
        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(train_loader):.4f}")
    losses.append(epoch_loss / len(train_loader))

torch.save(model.state_dict(), 'models/lstm_model.pth')

plt.plot(losses)
plt.title('LSTM Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('lstm_loss.png')
