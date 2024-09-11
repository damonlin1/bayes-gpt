# Bayesian GPT-2 Implementation

This repository contains an implementation of a GPT-2 model enhanced with Bayesian techniques using PyTorch and `torchbnn`. The notebook `bayes_gpt.ipynb` demonstrates the integration of Bayesian Neural Networks (BNN) with GPT-2 for uncertainty estimation and improved generalization.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Notebook Overview](#notebook-overview)
- [How to Run](#how-to-run)
- [Acknowledgments](#acknowledgments)

## Introduction

The purpose of this project is to explore the combination of generative models (specifically GPT-2) with Bayesian neural networks. This allows the model to estimate uncertainty in its predictions, providing more reliable outputs in scenarios where data is scarce or noisy.

## Requirements

To run the notebook, the following dependencies are required:

- Python 3.x
- `torch`
- `torchbnn` (Bayesian Neural Networks library for PyTorch)
- `transformers` (Transformers library)
- `pandas`
- `tqdm` (for progress bars)
- [Optional] Distributed training libraries from PyTorch for multi-GPU training

## Notebook Overview
The notebook is structured as follows:
1. Model Architecture:
   - A custom configuration class (GPT2Config) is defined to parameterize the GPT-2 model. Parameters include the number of layers, heads, vocabulary size, and prior distributions (mu, sigma).
2. Training Loop:
   - Distributed training using torch.distributed for scaling across multiple GPUs.
   - Integration with torchbnn to add Bayesian priors to GPT-2 weights.
3. Evaluation:
   - Methods for evaluating model uncertainty and performance metrics.

## How to Run
To execute the notebook:
1. Clone the repository:
```bash
git clone https://github.com/yourusername/bayesian-gpt.git
cd bayesian-gpt
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Open `bayes_gpt.ipynb` and run the cells sequentially to build and train the Bayesian GPT-2 model.

## Acknowledgments
This project uses GPT-2 from the Hugging Face transformers library and PyTorch's torchbnn for Bayesian Neural Networks.
