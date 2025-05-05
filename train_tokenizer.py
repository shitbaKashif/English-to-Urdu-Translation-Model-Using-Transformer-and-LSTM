from utils.tokenizer import BPETokenizer

# Function to load and preprocess the text data
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip().split('\n')

# Load your training data (English and Urdu)
en_train_data = load_data('data/bible/train.en')  # Adjust path as per your directory
ur_train_data = load_data('data/bible/train.ur')  # Adjust path as per your directory

# Initialize BPETokenizer for both English and Urdu
en_tokenizer = BPETokenizer(vocab_size=32000)  # Set vocabulary size for BPE
ur_tokenizer = BPETokenizer(vocab_size=32000)

# Train the tokenizers
print("Training English tokenizer...")
en_tokenizer.train(en_train_data)  # Train tokenizer on English data

print("Training Urdu tokenizer...")
ur_tokenizer.train(ur_train_data)  # Train tokenizer on Urdu data

# Save the trained tokenizers
print("Saving tokenizers...")
en_tokenizer.save('models/tokenizer_en.pt')  # Save English tokenizer
ur_tokenizer.save('models/tokenizer_ur.pt')  # Save Urdu tokenizer

print("Tokenizers saved as 'tokenizer_en.pt' and 'tokenizer_ur.pt'")
