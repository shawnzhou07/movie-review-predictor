# Movie Review Predictor

## Project Goal
Build a transformer language model from scratch and fine-tune it for movie review rating prediction (1-10 scale).

## Two-Phase Approach

### Phase 1: Pretrain Transformer (Language Modeling)
- Build decoder-only transformer architecture from scratch in PyTorch
- Implement multi-head self-attention, positional encoding, feed-forward layers
- Train on general text corpus (Wikipedia/OpenWebText, ~500M tokens)
- Task: Next-token prediction (language modeling)
- Target: ~25M parameters, achieve <50 perplexity

### Phase 2: Fine-tune for Rating Prediction
- Load pretrained transformer weights
- Replace language modeling head with regression head (1 output neuron)
- Fine-tune on IMDB movie reviews dataset (100k reviews)
- Task: Predict rating (1-10 scale) from review text
- Target: <0.8 star RMSE

**Fallback:** If rating prediction accuracy is poor, switch to 3-class sentiment classification (positive/negative/neutral)

## Tech Stack
- **Python** — primary language
- **PyTorch** — deep learning framework, custom nn.Module classes, autograd
- **NumPy** — data preprocessing, tokenization, metrics calculation
- **scikit-learn** — train/test split, evaluation metrics

## Project Structure
- `data/pretrain/` — general text corpus for pretraining
- `data/finetune/` — IMDB review dataset
- `src/` — source code (tokenizer, model architecture, training scripts)
- `models/` — saved model weights (pretrained.pt, finetuned.pt)
- `notebooks/` — Jupyter notebooks for analysis and visualization
- `outputs/` — generated predictions, metrics, plots

## Key Components to Implement
1. **Tokenizer** — Byte Pair Encoding (BPE) for subword tokenization
2. **Transformer Architecture** — Multi-head attention, positional encoding, feed-forward network, layer norm
3. **Pretraining Loop** — Language modeling with cross-entropy loss, Adam optimizer, learning rate scheduling
4. **Fine-tuning Loop** — Regression with MSE loss, lower learning rate, early stopping
5. **Inference** — Text generation (pretrained) and rating prediction (fine-tuned)

## Training Strategy
- Pretrain: 10-20 epochs on 500M tokens (~12-24 hours on GPU)
- Fine-tune: 5-10 epochs on 100k reviews (~2-4 hours)
- Hardware: Google Colab GPU (free tier) or local M4 MPS

## Known Challenges
- Memory constraints (use gradient accumulation, smaller batch sizes)
- Training stability (gradient clipping, learning rate scheduling)
- Data preprocessing (tokenization, padding, sequence length limits)
- Evaluation (perplexity for pretraining, RMSE for fine-tuning)

## Success Criteria
- Pretrained model generates coherent sentences
- Fine-tuned model predicts ratings with <0.8 star RMSE
- Complete pipeline from raw text → rating prediction