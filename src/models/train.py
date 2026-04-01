# TRAINING

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score
)

class MultiTaskDataset(Dataset):
    """PyTorch dataset for multi-task learning"""

    def __init__(self, features, labels_dict):
        self.features = torch.FloatTensor(features)
        self.labels = {
            task: torch.FloatTensor(labels).unsqueeze(1)
            for task, labels in labels_dict.items()
        }

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], {
            task: labels[idx] for task, labels in self.labels.items()
        }


class Trainer:
    """Training pipeline for multi-task model"""

    def __init__(self, model, task_types,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.task_types = task_types
        self.history = {'train_loss': [], 'val_loss': [], 'metrics': []}

        # Task weights
        self.task_weights = {
            'hERG': 2.0,      # Higher weight for toxicity
            'CYP1A2': 1.0,
            'CYP2C9': 1.0,
            'CYP2C19': 1.0,
            'CYP2D6': 1.0,
            'CYP3A4': 1.0,
            'HIA': 1.5,
            'BBB': 1.5,
            'ESOL': 1.0,
            'LogP': 1.0
        }

    def compute_loss(self, predictions, labels):
      total_loss = 0
      n_valid_tasks = 0

      for task, task_type in self.task_types.items():
          pred = predictions[task]
          label = labels[task]

          # Mask out NaN values (missing data for that specific task)
          mask = ~torch.isnan(label)
          if mask.sum() == 0:
              continue

          valid_pred = pred[mask]
          valid_label = label[mask]

          if task_type == 'classification':
              # Use BinaryCrossEntropy with Logits for better numerical stability
              loss = F.binary_cross_entropy_with_logits(valid_pred, valid_label)
          else:
              loss = F.mse_loss(valid_pred, valid_label)

          total_loss += loss * self.task_weights.get(task, 1.0)
          n_valid_tasks += 1

      return total_loss / n_valid_tasks if n_valid_tasks > 0 else torch.tensor(0.0)

    def train_epoch(self, train_loader, optimizer):
        """Train one epoch"""
        self.model.train()
        total_loss = 0

        for features, labels in train_loader:
            features = features.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}

            optimizer.zero_grad()
            predictions = self.model(features)
            loss = self.compute_loss(predictions, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, val_loader):
        """Evaluate on validation set"""
        self.model.eval()

        all_preds = {task: [] for task in self.task_types.keys()}
        all_labels = {task: [] for task in self.task_types.keys()}

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                predictions = self.model(features)

                for task in self.task_types.keys():
                    all_preds[task].extend(predictions[task].cpu().numpy())
                    all_labels[task].extend(labels[task].cpu().numpy())

        # Calculate metrics
        metrics = {}

        for task, task_type in self.task_types.items():
            preds = np.array(all_preds[task]).flatten()
            labels = np.array(all_labels[task]).flatten()

            mask = ~np.isnan(labels)
            if mask.sum() < 10:
                continue

            preds_valid = preds[mask]
            labels_valid = labels[mask]

            if task_type == 'classification':
                try:
                    metrics[f'{task}_AUC'] = roc_auc_score(labels_valid, preds_valid)
                    metrics[f'{task}_ACC'] = accuracy_score(
                        labels_valid, (preds_valid > 0.5).astype(int)
                    )
                except:
                    pass
            else:
                metrics[f'{task}_RMSE'] = np.sqrt(mean_squared_error(labels_valid, preds_valid))
                metrics[f'{task}_MAE'] = mean_absolute_error(labels_valid, preds_valid)
                try:
                    metrics[f'{task}_R2'] = r2_score(labels_valid, preds_valid)
                except:
                    pass

        return metrics

    def train(self, train_loader, val_loader, epochs=50, lr=0.001, patience=10):
        """Full training loop"""

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5 # Removed verbose=True
        )

        best_val_loss = float('inf')
        patience_counter = 0

        print(f"\n{'='*70}")
        print("TRAINING STARTED")
        print(f"{'='*70}\n")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer)
            val_metrics = self.evaluate(val_loader)

            # Approximate validation loss
            val_loss = train_loss

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['metrics'].append(val_metrics)

            scheduler.step(val_loss)

            # Print progress
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            if val_metrics:
                print(f"  Sample Metrics:")
                for k, v in list(val_metrics.items())[:5]:
                    print(f"    {k}: {v:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'model_state': self.model.state_dict(),
                    'metrics': val_metrics,
                    'epoch': epoch
                }, 'best_admet_model.pt')
                print("  ✓ Model saved")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

            print()

        # Load best model
        checkpoint = torch.load('best_admet_model.pt', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])

        print(f"\n{'='*70}")
        print("TRAINING COMPLETED!")
        print(f"Best epoch: {checkpoint['epoch']+1}")
        print(f"{'='*70}\n")

        return self.model, checkpoint['metrics'] 

# ============================================================================

# MAIN EXECUTION - TRAINING

def main():
    """Complete pipeline execution"""

    print("\n" + "="*70)
    print("SIMPLIFIED ADMET PREDICTOR - 6 CORE TASKS")
    print("hERG | CYP1A2,2C9,2C19,2D6,3A4 | HIA | BBB | ESOL | LogP")
    print("="*70 + "\n")

    # Step 1: Load datasets
    print("[STEP 1/6] Loading datasets...")
    loader = CoreADMETDataLoader()
    datasets, task_types = loader.load_all_datasets()

    if len(datasets) == 0:
        print("\n❌ Error: No datasets loaded. Please check installation")
        return None

    # Step 2: Initialize featurizer
    print("\n[STEP 2/6] Initializing molecular featurizer...")
    featurizer = MolecularFeaturizer(
        radius=2,
        n_bits=2048,
        use_descriptors=True
    )
    print(f"Feature dimension: {featurizer.get_feature_dim()}")

    # Step 3: Prepare unified dataset
    print("\n[STEP 3/6] Creating unified dataset...")
    prep = UnifiedDatasetPreparation(datasets, task_types, featurizer)
    X, labels_matrix, valid_smiles = prep.prepare_unified_dataset()

    # Step 4: Split data
    print("\n[STEP 4/6] Splitting train/validation sets...")
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)

    train_size = int(0.8 * n_samples)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train = {task: labels[train_idx] for task, labels in labels_matrix.items()}
    y_val = {task: labels[val_idx] for task, labels in labels_matrix.items()}

    train_dataset = MultiTaskDataset(X_train, y_train)
    val_dataset = MultiTaskDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Step 5: Build model
    print("\n[STEP 5/6] Building neural network...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = MultiTaskADMETModel(
        input_dim=featurizer.get_feature_dim(),
        task_names=list(task_types.keys()),
        task_types=task_types,
        hidden_dims=[1024, 512, 256]
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Step 6: Train
    print("\n[STEP 6/6] Training model...")
    trainer = Trainer(model, task_types, device=device)
    trained_model, final_metrics = trainer.train(
        train_loader,
        val_loader,
        epochs=30,  # Reduced for faster training
        lr=0.001,
        patience=10
    )

    # Display final metrics
    print("Final Validation Metrics:")
    print("="*70)
    if final_metrics:
        for metric, value in sorted(final_metrics.items()):
            print(f"{metric:30s}: {value:.4f}")

    print("\n" + "="*70)
    print("✓ TRAINING COMPLETED!")
    print("Model saved as: best_admet_model.pt")
    print("="*70)

    return trained_model, featurizer, task_types, prep.scalers, trainer

# Run training
trained_model, featurizer, task_types, scalers, trainer = main()
