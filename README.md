# 🧠 Wikipedia Title Generation using Seq2Seq and Transformers

## 📌 Overview
This project focuses on generating **Wikipedia article titles** from their **body text** using multiple sequence-to-sequence modeling techniques. It involves building and evaluating:
- Basic RNN-based encoder-decoder models
- Hierarchical & dual-layer RNNs
- Beam search decoding
- Pre-trained Transformer models (T5 and Flan-T5)
- Prompt engineering for zero-shot inference

> 💡 Built as part of CS60075 NLP Assignment 2 (Spring 2025), IIT Kharagpur.

---

## 📁 Project Structure

| Part | Title                                  | Description |
|------|----------------------------------------|-------------|
| A    | Dataset Preprocessing                  | Clean and tokenize Wikipedia dataset |
| B1   | Basic RNN Seq2Seq                      | EncoderRNN + DecoderRNN model from scratch |
| B2   | RNN Model Enhancements                 | GloVe embeddings, Hierarchical RNNs, Dual-decoder, Beam Search |
| C1   | Transformer Fine-Tuning                | T5-small fine-tuned with HuggingFace Seq2SeqTrainer |
| C2   | Prompt-based Zero-shot Inference       | Flan-T5 models with prompt variations |

---

## 🛠️ Technologies Used

- **PyTorch** & **TorchText**
- **HuggingFace Transformers**
- **NLTK / SpaCy**
- **GloVe Embeddings**
- **ROUGE Scores** (1, 2, L-F1)
- **Google Colab** / GPU

---

## 🧪 Tasks Implemented

### ✅ Part A: Dataset and Preprocessing
- Text cleaning (punctuation, non-ASCII removal)
- Stopword removal, stemming/lemmatization
- Splitting training/validation/test sets

### ✅ Part B1: Seq2Seq with RNN
- `EncoderRNN`, `DecoderRNN`, `Seq2seqRNN` classes
- Training with teacher-forcing
- Greedy decoding during inference
- ROUGE evaluation on generated titles

### ✅ Part B2: Model Improvements
- GloVe embedding integration via `load_embeddings`
- `HierEncoderRNN`: sentence-level GRU over word-level embeddings
- `Decoder2RNN`: dual-layer decoder GRUs
- Beam Search decoding with top-k tracking

### ✅ Part C1: Transformer Fine-tuning
- Pretrained `google-t5/t5-small` using `AutoModelForSeq2SeqLM`
- Fine-tuning using `Seq2SeqTrainer` on the Wikipedia dataset
- Greedy and Beam search decoding

### ✅ Part C2: Prompt Engineering
- `google/flan-t5-base` and `google/flan-t5-large` for zero-shot prediction
- Prompt variations tested to optimize results without training

---
