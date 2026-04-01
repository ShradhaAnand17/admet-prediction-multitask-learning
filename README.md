# ADMET Multi-Task Deep Learning

Multi-task deep learning model for predicting ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties from molecular structures using shared representations across heterogeneous datasets.

---

## Overview

This project implements an end-to-end pipeline for ADMET prediction using a multi-task neural network. Instead of training separate models for each property, the model learns shared molecular representations to improve generalization across tasks.

---

## Key Features

- Multi-task learning across multiple ADMET endpoints
- Molecular featurization from SMILES (RDKit-based)
- Unified dataset pipeline from multiple sources
- Joint training for classification and regression tasks
- Scalable PyTorch-based architecture
- Inference pipeline for new molecules
- Flask-based deployment (optional)

---

## 🏗️ Project Structure

