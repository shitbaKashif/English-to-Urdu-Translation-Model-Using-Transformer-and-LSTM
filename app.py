import streamlit as st
import torch
from models.transformer import TransformerModel
from models.lstm import LSTMTranslator
from utils.tokenizer import BPETokenizer  # Import the BPETokenizer
from utils.evaluate import evaluate  # For evaluation
from utils.dataset import TranslationDataset  # For loading datasets
from utils.visualization import VizAttention  # For attention visualization

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Tokenizers
src_tokenizer = BPETokenizer()
src_tokenizer.load('models/tokenizer_en.pt')  # Load English tokenizer
tgt_tokenizer = BPETokenizer()
tgt_tokenizer.load('models/tokenizer_ur.pt')  # Load Urdu tokenizer

# Load pre-trained Transformer model
transformer = TransformerModel(
    src_vocab_size=src_tokenizer.vocab_size,
    tgt_vocab_size=tgt_tokenizer.vocab_size
).to(device)
transformer.load_state_dict(torch.load('models/transformer_model.pth'))
transformer.eval()

# Load pre-trained LSTM model
lstm = LSTMTranslator(
    src_vocab_size=src_tokenizer.vocab_size,
    tgt_vocab_size=tgt_tokenizer.vocab_size
).to(device)
lstm.load_state_dict(torch.load('models/lstm_model.pth'))
lstm.eval()

# Streamlit interface
st.title("English to Urdu Translation")

history = []

input_text = st.text_input("Enter English Text:")

if input_text:
    # Tokenize the input sentence
    src = torch.tensor([src_tokenizer.tokenize(input_text)]).to(device)
    tgt_input = torch.zeros((1, 1), dtype=torch.long).to(device)

    outputs = []
    attention_weights = []
    with torch.no_grad():
        for _ in range(50):
            output = transformer(src, tgt_input)  # Use the pre-trained transformer model
            pred_token = output[:, -1, :].argmax(-1)
            outputs.append(pred_token.item())

            # Capture attention weights from the model (if available)
            if hasattr(transformer, 'get_attention_weights'):
                attn_weights = transformer.get_attention_weights(src, tgt_input)
                attention_weights.append(attn_weights)

            if pred_token.item() == 0:
                break
            tgt_input = torch.cat([tgt_input, pred_token.unsqueeze(1)], dim=1)

    # Convert token IDs to text
    translated = tgt_tokenizer.detokenize(outputs)

    # Debugging: Print the translated tokens to check if they are correct
    print("Translated tokens:", outputs)
    print("Translated text:", translated)

    history.append((input_text, translated))

    # Show the conversation history
    for src, tgt in history:
        st.write(f"**ENGLISH:** {src}")
        st.markdown(f"<div style='text-align: right; font-family: 'Noto Nastaliq Urdu', serif;'>**URDU:** {tgt}</div>", unsafe_allow_html=True)
        st.markdown("---")

    # Visualize Attention when clicked
    if st.button("Visualize Attention"):
        if attention_weights:
            fig = VizAttention(attention_weights[0], input_text, translated)  # Visualize the attention for the first prediction
            st.pyplot(fig)
        else:
            st.write("No attention weights available.")

# Add Evaluation Metrics (BLEU & ROUGE) for the models
if st.button("Evaluate Model"):
    test_dataset = TranslationDataset('data/bible/test.en', 'data/bible/test.ur', src_tokenizer, tgt_tokenizer)
    
    # Evaluate the Transformer and LSTM models using the evaluate function from evaluate.py
    bleu_score_transformer, rouge_score_transformer = evaluate(transformer, test_dataset, src_tokenizer, tgt_tokenizer)
    bleu_score_lstm, rouge_score_lstm = evaluate(lstm, test_dataset, src_tokenizer, tgt_tokenizer)

    st.write(f"Transformer BLEU Score: {bleu_score_transformer}")
    st.write(f"Transformer ROUGE Score: {rouge_score_transformer}")
    st.write(f"LSTM BLEU Score: {bleu_score_lstm}")
    st.write(f"LSTM ROUGE Score: {rouge_score_lstm}")
