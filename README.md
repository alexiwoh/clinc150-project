# CLINC150 Intent Classification and Out-of-Scope Detection
Deep Learning for Intent Classification and Out-of-Scope Detection: A Comparative Study of Lightweight Neural Architectures

This project explores how lightweight deep learning models can be used to understand assistant-style user queries. The main goal is to compare multiple neural network architectures on the task of **intent classification** and **out-of-scope (OOS) detection** using the **CLINC150** dataset.

CLINC150 is a public benchmark dataset containing short natural-language queries across many intent classes, along with out-of-scope examples that do not belong to any supported intent. This makes it a strong fit for studying problems related to AI assistants, query routing, tool selection, and guardrails.

The project is implemented in **Python with PyTorch** and compares three model families:

- **TF-IDF + MLP baseline**
- **Text CNN**
- **BiLSTM**

The purpose of the project is not only to measure raw classification accuracy, but also to study how well these models can distinguish valid in-scope intents from unknown or unsupported queries. This mirrors real-world AI engineering problems where systems must both route known requests correctly and avoid confidently mishandling unknown ones.

The final output of the project includes:

- training and validation loss curves
- test accuracy and macro F1 score
- out-of-scope precision, recall, and F1
- confusion matrix for the best-performing model
- comparison of model size, training time, and common error patterns

Overall, this project provides a practical introduction to deep learning for NLP while staying closely connected to real-world assistant and agent systems.