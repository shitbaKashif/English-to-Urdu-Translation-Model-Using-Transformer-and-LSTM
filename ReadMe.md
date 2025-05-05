Here is a **detailed README** that you can use for your project. It outlines the entire process, from setup to running the app, and provides an overview of the project's components.

---

# **English to Urdu Translation Model Using Transformer and LSTM**

This project demonstrates the implementation of machine translation from **English to Urdu** using two models:
- **Transformer Model**
- **LSTM Model**

The translation models are trained on the **UMC005 English-Urdu Parallel Corpus**, and the system allows users to input English sentences and get translated Urdu sentences. Additionally, the project includes:
- Evaluation metrics (BLEU, ROUGE)
- Attention visualization for the Transformer model
- Pre-trained models that can be used for translation without retraining

## **Project Structure**

The project is structured as follows:

```
/Q1
├── data/
│   ├── bible/
│   ├── quran/
├── models/
│   ├── transformer.py
│   ├── lstm.py
├── utils/
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── metrics.py
│   ├── visualization.py
├── app.py
├── train_transformer.py
├── train_lstm.py
├── evaluate.py
├── requirements.txt
├── README.md
└── saved_models/
```

### **Key Files and Folders**:
- **`data/`**: Contains the English and Urdu sentences for training, validation, and testing.
- **`models/`**: Contains model implementations for both Transformer (`transformer.py`) and LSTM (`lstm.py`).
- **`utils/`**: Contains utility functions:
  - **`tokenizer.py`**: Tokenizer implementation for Byte Pair Encoding (BPE).
  - **`dataset.py`**: Dataset class for loading and processing data.
  - **`metrics.py`**: Functions for calculating evaluation metrics like BLEU and ROUGE.
  - **`visualization.py`**: Function for visualizing attention weights.
- **`app.py`**: Streamlit interface for inputting text and getting translations, along with model evaluation and attention visualization.
- **`train_transformer.py`**: Script to train the Transformer model.
- **`train_lstm.py`**: Script to train the LSTM model.
- **`evaluate.py`**: Script to evaluate the models on the test dataset using BLEU and ROUGE metrics.

---

## **Installation and Setup**

### **1. Install Required Libraries**

Before running the project, you need to install the required dependencies. The project uses **PyTorch**, **Streamlit**, and several other libraries.

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

### **2. Download the Data**

You will need to download the **UMC005 English-Urdu Parallel Corpus** for training and evaluation. This corpus is available from the official site. You can place the downloaded data in the `data/` directory.

After downloading, the folder structure will look like:

```
/data
├── bible
│   ├── train.en
│   ├── train.ur
│   ├── dev.en
│   ├── dev.ur
│   ├── test.en
│   ├── test.ur
├── quran
│   ├── train.en
│   ├── train.ur
│   ├── dev.en
│   ├── dev.ur
│   ├── test.en
│   ├── test.ur
```

### **3. Pre-trained Models**

You can use the pre-trained models stored in the `models/` directory:
- **`tokenizer_en.pt`**: English tokenizer
- **`tokenizer_ur.pt`**: Urdu tokenizer
- **`transformer_model.pth`**: Pre-trained Transformer model
- **`lstm_model.pth`**: Pre-trained LSTM model

These models are loaded in the `app.py` file for translation purposes. If you don't have these models, you can train them using the provided `train_transformer.py` and `train_lstm.py` scripts.

---

## **Running the App**

### **1. Launching the Streamlit App**

The core of this project is the **Streamlit interface** (`app.py`), which allows users to input English sentences and get their corresponding Urdu translations.

To run the app:

```bash
streamlit run app.py
```

This will open a web browser with the interface where you can:
- Enter English text to get the Urdu translation.
- Visualize the attention weights for the Transformer model.
- Evaluate the models on the test set and display BLEU and ROUGE scores.

### **2. Training the Models**

If you want to train the **Transformer** or **LSTM** models from scratch, you can run the following scripts:

- **Training the Transformer model**:

  ```bash
  python train_transformer.py
  ```

- **Training the LSTM model**:

  ```bash
  python train_lstm.py
  ```

This will train the models on the UMC005 English-Urdu dataset and save the trained models in the `saved_models/` directory.

### **3. Evaluating the Models**

You can evaluate the models' performance (on BLEU and ROUGE metrics) by running the `evaluate.py` script:

```bash
python evaluate.py
```

This will evaluate both the **Transformer** and **LSTM** models on the test dataset and print the BLEU and ROUGE scores.

### **4. Visualizing Attention**

In the **Streamlit interface** (`app.py`), you can click the **"Visualize Attention"** button to see the attention heatmap for the Transformer model. This shows which parts of the input sentence the model focused on while generating the translation.

---

## **Evaluation Metrics**

### **1. BLEU Score**

The **BLEU (Bilingual Evaluation Understudy)** score measures the precision of n-grams (typically unigram, bigram, trigram) between the predicted and reference translations. A higher BLEU score indicates better translation quality.

### **2. ROUGE Score**

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** is another metric that evaluates the overlap between n-grams in the predicted and reference sentences. ROUGE-1, ROUGE-2, and ROUGE-L scores are commonly used to evaluate translation quality.

Both **BLEU** and **ROUGE** scores are computed in the `evaluate.py` and displayed on the Streamlit interface.

---

## **Attention Visualization**

The **Transformer model** includes an attention mechanism, which helps the model decide which parts of the input to focus on during translation. The attention weights can be visualized as a heatmap, showing the alignment between words in the input and output sentences. This is done using the **`VizAttention()`** function in the `utils/visualization.py` file.

### Example of Attention Visualization:

- **Input**: "I love books."
- **Output**: "مجھے کتابیں پسند ہیں۔"
- The heatmap will show how much attention was given to each word in the input sentence when generating each word in the output sentence.

---

## **Future Improvements**

### **1. Fine-tuning Pre-trained Models**
Currently, the models are trained from scratch. However, fine-tuning **pre-trained models** (like **BERT** or **GPT**) on the English-Urdu dataset might improve performance. You could experiment with fine-tuning these models using **transfer learning**.

### **2. Hyperparameter Tuning**
Although hyperparameters like learning rate, batch size, and model depth are set, further **hyperparameter tuning** using techniques like **Grid Search** or **Bayesian Optimization** could yield better results.

### **3. Multi-Lingual Models**
This project focuses on **English to Urdu** translation. However, you could extend it to **multi-lingual translation** by training on additional language pairs.

---

## **Conclusion**

This project demonstrates the implementation of machine translation from **English to Urdu** using both **Transformer** and **LSTM** models. It provides a **Streamlit-based user interface** for easy text input and translation, along with model evaluation and attention visualization. You can either use pre-trained models or train your own on the **UMC005 English-Urdu Parallel Corpus**.

Feel free to experiment with the models, fine-tune them, and extend the project for additional language pairs and features!

---

### **Acknowledgments**
- The project uses the **UMC005 English-Urdu Parallel Corpus** for training and evaluation.
- The models are based on the Transformer architecture as described in the paper **"Attention is All You Need"** and the LSTM-based sequence-to-sequence model.

---

### **License**

This project is open-source and free for non-commercial educational and research purposes. For commercial use, please reach out to the authors for licensing details.

