# Task 2: LSTM & Transformer Chatbots — Plan

## Topic: Cooking Q&A

---

## 1. Dataset Preparation (~1 point)

- **Source**: Use an open cooking/recipe Q&A dataset (e.g., RecipeQA, or filtered cooking subset from large-qa-datasets).
- **Size**: Minimum 2,000 question-answer pairs in English.
- **Preprocessing**:
  - Lowercase, remove special characters, basic tokenization.
  - Build vocabulary (word-to-index mapping) with `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>` tokens.
  - Limit vocabulary size (e.g., top 10K words).
  - Pad/truncate sequences to fixed max length.
  - Split: 80% train / 10% validation / 10% test.

## 2. Model Implementation & Training (~2 points)

### 2.1 LSTM Seq2Seq Model
- **Architecture**: Encoder-Decoder with LSTM cells.
  - Encoder: Embedding → LSTM (1–2 layers, hidden size ~256).
  - Decoder: Embedding → LSTM → Linear → Softmax.
  - Teacher forcing during training.
- **Training**: Cross-entropy loss, Adam optimizer, ~30–50 epochs, batch size 64.

### 2.2 Transformer Seq2Seq Model
- **Architecture**: Custom Transformer (built from scratch, no pretrained weights).
  - Positional encoding (sinusoidal).
  - Encoder: 2–4 layers, multi-head self-attention + feed-forward.
  - Decoder: 2–4 layers, masked self-attention + cross-attention + feed-forward.
  - Embedding size ~256, 4 attention heads.
- **Training**: Cross-entropy loss, Adam with warm-up scheduler, ~30–50 epochs.

### Common
- Framework: PyTorch.
- Greedy decoding (and optionally beam search) for inference.
- Track training/validation loss per epoch, plot learning curves.

## 3. Results Analysis & Evaluation (~1 point)

- Select **at least 10 test questions** on cooking topics.
- Generate answers from both models for each question.
- Present results in a comparison table:
  | # | Question | LSTM Answer | Transformer Answer |
  |---|----------|-------------|--------------------|
  | 1 | ...      | ...         | ...                |
- **Metrics**: BLEU score on test set for both models.
- **Qualitative analysis**: Discuss fluency, relevance, and common failure modes.
- **Conclusions**: Compare LSTM vs Transformer performance, strengths, and weaknesses.

---

## Deliverables

1. `PLAN.md` — this file.
2. Jupyter notebook(s) with full code (data prep, training, evaluation).
3. Trained model checkpoints (optional).
4. Report section with model descriptions, training plots, comparison table, and conclusions.
