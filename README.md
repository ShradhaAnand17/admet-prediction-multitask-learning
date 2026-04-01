# ADMET Multi-Task Deep Learning

Multi-task deep learning model for predicting ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties from molecular structures using shared representations across heterogeneous datasets.

## Overview

This project implements an end-to-end pipeline for ADMET prediction using a multi-task neural network. Instead of training separate models for each property, the model learns shared molecular representations to improve generalization across tasks.

## Key Features

- Multi-task learning across multiple ADMET endpoints
- Molecular featurization from SMILES (RDKit-based)
- Unified dataset pipeline from multiple sources
- Joint training for classification and regression tasks
- Scalable PyTorch-based architecture
- Inference pipeline for new molecules
- Flask-based deployment

## Model Architecture

- Shared feature extractor (fully connected layers)
- Task-specific heads:
  - Classification (hERG toxicity, CYP450 Inhibition, BBB permeability, HIA)
  - Regression (solubility, lipophilicity)
- Loss functions:
  - Binary Cross Entropy (BCE) for classification
  - Mean Squared Error (MSE) for regression

## Evaluation Metrics
- **Classification:**
  - ROC-AUC
  - Accuracy
- **Regression:**
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)

## Web App (Colab-Based Deployment)
An interactive Flask-based interface is included for real-time ADMET prediction. The web app is deployed using ngrok inside Google Colab
- The app is served via a temporary public URL generated during runtime (Colab + tunneling)
- The server runs only while the Colab notebook is active. 
- This allows quick testing without full deployment
> Note: If the runtime disconnects or the cell stops, the app goes offline. The link expires once the session ends.

## Tech Stack
- Python
- PyTorch
- RDKit
- Scikit-learn
- Flask

