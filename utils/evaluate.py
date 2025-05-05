import torch
from utils.tokenizer import BPETokenizer  # Updated import
from utils.dataset import TranslationDataset
from utils.metrics import compute_bleu, compute_rouge
from models.transformer import TransformerModel
from models.lstm import LSTMTranslator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, dataset, src_tokenizer, tgt_tokenizer):
    model.eval()
    references, hypotheses = [], []

    with torch.no_grad():
        for src, tgt in dataset:
            src = src.unsqueeze(0).to(device)
            tgt = tgt.unsqueeze(0).to(device)
            tgt_input = tgt[:, :-1]

            output = model(src, tgt_input)
            output = output.argmax(-1)

            ref = tgt_tokenizer.detokenize(tgt.squeeze(0).tolist())  # Updated to detokenize using BPE tokenizer
            hyp = tgt_tokenizer.detokenize(output.squeeze(0).tolist())  # Updated to detokenize using BPE tokenizer

            references.append(ref)
            hypotheses.append(hyp)
    
    bleu = compute_bleu(references, hypotheses)
    rouge = compute_rouge(references, hypotheses)
    return bleu, rouge

if __name__ == "__main__":
    # Load Tokenizers
    src_tokenizer = BPETokenizer()
    src_tokenizer.load('models/tokenizer_en.pt')  # Load English tokenizer
    tgt_tokenizer = BPETokenizer()
    tgt_tokenizer.load('models/tokenizer_ur.pt')  # Load Urdu tokenizer

    # Load Test Dataset
    test_dataset = TranslationDataset('data/bible/test.en', 'data/bible/test.ur', src_tokenizer, tgt_tokenizer)

    # Load Transformer model
    transformer = TransformerModel(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size
    ).to(device)
    transformer.load_state_dict(torch.load('models/transformer_model.pth'))

    # Load LSTM model
    lstm = LSTMTranslator(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size
    ).to(device)
    lstm.load_state_dict(torch.load('models/lstm_model.pth'))

    # Evaluate models
    bleu_trans, rouge_trans = evaluate(transformer, test_dataset, src_tokenizer, tgt_tokenizer)
    bleu_lstm, rouge_lstm = evaluate(lstm, test_dataset, src_tokenizer, tgt_tokenizer)

    # Print results
    print(f"Transformer BLEU: {bleu_trans:.4f}")
    print(f"LSTM BLEU: {bleu_lstm:.4f}")
    print(f"Transformer ROUGE: {rouge_trans}")
    print(f"LSTM ROUGE: {rouge_lstm}")
