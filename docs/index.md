---
title: Home
hide:
  - navigation
  - toc
---

<p align="center"> <img src="_assets/logo.svg" alt="Logo" width="120em" /> </p>

<h1 align="center" style="margin: 0;"> Yelp Review Sentiment </h1>

<h3 align="center" style="margin: 0.6em;">
    Classifying Yelp review polarity with scratch PyTorch NLP models.
</h3>

<h3 align="center" style="margin: 1em; margin-bottom: 3em;">
    <a href="https://oguzhanozkaya.github.io/yelp-review-sentiment/">Medium Article</a> - <a href="https://oguzhanozkaya.github.io/yelp-review-sentiment/">Presentation</a>
</h3>

<div class="grid cards" markdown>

-   ## Project

    This project predicts whether a Yelp review is negative or positive from review text. The implementation uses raw PyTorch only: tokenization, vocabulary construction, batching, the neural architecture, optimization, evaluation, and artifact generation are implemented in `yrs.py` without fastai, pretrained embeddings, pretrained language models, or transformer libraries.

    **Dataset**: [Yelp Review Dataset](https://www.kaggle.com/datasets/ilhamfp31/yelp-review-dataset)

    **Student**: Oğuzhan Özkaya

    **Instructor**: Şafak Özden

    _ADA 447 Introduction to Deep Learning - TED University_

</div>

<div class="grid cards" markdown>

-   ## Objective

    Build a reproducible command-driven sentiment classifier that reads the local Yelp CSV files on each run, trains a high-capacity scratch TextCNN, evaluates validation and test performance, and generates report-ready outputs.

-   ## Approach

    Convert review text into token ids with an in-memory training vocabulary, apply word dropout as text augmentation, and train a Kim-style convolutional neural network over learned word embeddings. The model uses multiple convolution widths to capture short sentiment phrases and pools the strongest signals for binary classification.

</div>
