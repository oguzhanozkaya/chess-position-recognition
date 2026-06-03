# Yelp Review Sentiment with Scratch PyTorch TextCNN

## Abstract

This project predicts Yelp review polarity with a reproducible deep learning pipeline. The target is binary sentiment: negative or positive. The model reads raw review text, builds a vocabulary from the active training split, and trains a scratch PyTorch TextCNN without fastai, pretrained embeddings, pretrained language models, transformer libraries, or language model APIs.

Final result placeholders in this article should be filled after the extended RTX 4080 training run.

## Problem Definition

Each row represents one Yelp review. The model receives the review text and predicts whether the original polarity label is negative or positive.

| Raw Label | Class      |
| --------- | ---------- |
| `1`       | `negative` |
| `2`       | `positive` |

## Data

Raw data comes from the [Yelp Review Dataset](https://www.kaggle.com/datasets/ilhamfp31/yelp-review-dataset). The files are placed locally before running the pipeline:

- `data/train.csv` for training and validation.
- `data/test.csv` for final held-out evaluation.

Both files are headerless CSV files with label in the first column and review text in the second column. The pipeline reads these files on every run. No processed dataset cache is written.

## Text Processing

The pipeline builds the text representation in memory:

- lowercase regex tokenization extracts words, simple contractions, digits, and punctuation;
- the vocabulary is built only from the training split;
- rare and unseen tokens map to `<unk>`;
- reviews are padded or truncated to a fixed token length;
- training-time word dropout randomly replaces non-padding tokens with `<unk>` as augmentation.

## Model

Only one active architecture is trained: a scratch TextCNN classifier implemented directly with PyTorch.

For each review:

- token ids are mapped to learned word embeddings;
- parallel 1D convolution filters with multiple widths detect local sentiment phrases;
- global max pooling keeps the strongest activation from each filter bank;
- a dropout-regularized MLP predicts negative/positive logits.

This architecture is appropriate for the course scope because it exercises natural language processing, convolutional neural networks, data preprocessing, augmentation, optimization, and end-to-end PyTorch implementation without relying on fastai or pretrained models.

## Evaluation

The training CSV is split into stratified train and validation partitions. The test CSV remains isolated until final evaluation.

Metrics:

- Accuracy measures overall correct predictions.
- Macro F1 measures class-balanced quality.
- Log loss measures probability quality.
- Per-class precision, recall, and F1 show class-specific behavior.
- Confusion matrix shows error structure.

## Results

Replace this section after the extended training run.

| Split      | Accuracy | Macro F1 | Log Loss |
| ---------- | -------- | -------- | -------- |
| Validation | TODO     | TODO     | TODO     |
| Test       | TODO     | TODO     | TODO     |

Generated figures after `just run`:

- `output/figures/confusion_matrix.png`
- `output/figures/class_distribution.png`
- `output/figures/prediction_confidence.png`
- `output/figures/training_loss.png`

## Reproducibility

Run the full pipeline:

```bash
just sync
just run
```

Short CPU smoke run:

```bash
just smoke
```

## Limitations

The model uses a word-level vocabulary, so rare spelling variants and unusual tokens can map to `<unk>`. Long reviews are truncated to the configured maximum sequence length. Final report quality depends on the extended local GPU training run and may improve with careful tuning of vocabulary size, sequence length, dropout, and convolution width.
