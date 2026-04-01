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
The model is built using PyTorch and follows a shared backbone design:
- Shared Feature Extractor
  - Fully connected neural network layers
  - Learns generalized molecular embeddings
- Task-Specific Heads
  - Classification Tasks:
    - hERG toxicity
    - CYP450 inhibition
    - Blood-Brain Barrier (BBB) permeability
    - Human Intestinal Absorption (HIA)
  - Regression Tasks:
    - Aqueous solubility (ESOL)
    - Lipophilicity (LogP)
- Loss Functions
  - Binary Cross Entropy (BCE) for classification
  - Mean Squared Error (MSE) for regression

## Molecular Featurization
Molecular structures are processed using RDKit:
- SMILES parsing
- Molecular descriptors
- Fingerprint generation

## Evaluation Metrics
- **Classification:**
  - ROC-AUC
  - Accuracy
- **Regression:**
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
 
## ⚙️ Pipeline Overview
SMILES → Featurization → Shared Neural Network → Task-specific Outputs → Predictions

## Web App (Colab-Based Deployment)
An interactive Flask-based interface is included for real-time ADMET prediction. The web app is deployed using ngrok inside Google Colab
- The app is served via a temporary public URL generated during runtime (Colab + tunneling)
- The server runs only while the Colab notebook is active. 
- This allows quick testing without full deployment
> Note: If the runtime disconnects or the cell stops, the app goes offline. The link expires once the session ends.

## 🛠️ Tech Stack
- Python  
- PyTorch  
- RDKit  
- Scikit-learn  
- Flask  

## ⚠️ Limitations
- Requires an active Google Colab runtime  
- Generates temporary (non-persistent) deployment URLs  
- Dependent on tunneling service stability  

## Tech Stack
- Python
- PyTorch
- RDKit
- Scikit-learn
- Flask

## Limitations
- Requires active Colab runtime
- Temporary deployment URL
- Dependent on tunneling service stability

## Applications
- Drug discovery and screening
- ADMET risk assessment
- Computational pharmacology
- AI-driven medicinal chemistry

