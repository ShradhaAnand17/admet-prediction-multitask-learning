# ============================================================================
# UNIFIED DATASET PREPARATION
# ============================================================================

class UnifiedDatasetPreparation:
    """Prepare unified dataset with all molecules across tasks"""
    def __init__(self, datasets_dict, task_types, featurizer):
        self.datasets = datasets_dict
        self.task_types = task_types
        self.featurizer = featurizer
        self.scalers = {}

    def prepare_unified_dataset(self):
        print("Preparing unified multi-task dataset...")

        # 1. Collect all unique SMILES first
        all_smiles_raw = set()
        for dataset_name, df in self.datasets.items():
            all_smiles_raw.update(df['Drug'].dropna().unique())

        # 2. Featurize and only keep molecules that pass RDKit validation
        features_list = []
        valid_smiles = []

        print(f"Featurizing {len(all_smiles_raw)} unique molecules...")
        for smiles in list(all_smiles_raw):
            feats = self.featurizer.featurize(smiles)
            if feats is not None:
                features_list.append(feats)
                valid_smiles.append(smiles)

        X = np.array(features_list)
        n_mols = len(valid_smiles)
        smiles_to_idx = {s: i for i, s in enumerate(valid_smiles)}

        # 3. Create Label Matrix with NaNs for missing data
        labels_matrix = {task: np.full(n_mols, np.nan) for task in self.task_types.keys()}

        for task_name, df in self.datasets.items():
            for _, row in df.iterrows():
                s = row['Drug']
                if s in smiles_to_idx:
                    idx = smiles_to_idx[s]
                    labels_matrix[task_name][idx] = row['Y']

        # 4. Scale Regression Tasks (ESOL and LogP)
        for task, t_type in self.task_types.items():
            if t_type == 'regression':
                labels = labels_matrix[task]
                mask = ~np.isnan(labels)
                if mask.any():
                    scaler = StandardScaler()
                    labels[mask] = scaler.fit_transform(labels[mask].reshape(-1, 1)).flatten()
                    self.scalers[task] = scaler

        print(f"Dataset ready: {X.shape[0]} molecules, {X.shape[1]} features.")
        return X, labels_matrix, valid_smiles
