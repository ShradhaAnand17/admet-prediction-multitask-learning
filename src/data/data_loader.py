class CoreADMETDataLoader:
    """
    Load 6 core ADMET datasets:
    - hERG (cardiotoxicity)
    - CYP 1A2, 2C9, 2C19, 2D6, 3A4 (drug-drug interactions)
    - HIA (absorption)
    - BBB (brain penetration)
    - ESOL (solubility)
    - LogP (lipophilicity)
    """

    def __init__(self):
        self.datasets = {}
        self.task_types = {}

    def load_herg_from_tdc(self):
        """Load hERG cardiotoxicity from TDC"""
        print("Loading hERG (Cardiotoxicity)...")
        try:
            data = Tox(name='hERG')
            df = data.get_data()
            self.datasets['hERG'] = df
            self.task_types['hERG'] = 'classification'
            print(f"✓ hERG: {len(df)} compounds")
            return True
        except Exception as e:
            print(f"✗ hERG failed: {e}")
            return False

    def load_cyp_from_tdc(self):
        """Load CYP inhibition from TDC"""
        print("\nLoading CYP Inhibition Assays...")
        cyp_enzymes = ['1A2', '2C9', '2C19', '2D6', '3A4']

        for cyp in cyp_enzymes:
            try:
                data = ADME(name=f'CYP{cyp}_Veith')
                df = data.get_data()
                self.datasets[f'CYP{cyp}'] = df
                self.task_types[f'CYP{cyp}'] = 'classification'
                print(f"✓ CYP{cyp}: {len(df)} compounds")
            except Exception as e:
                print(f"✗ CYP{cyp} failed: {e}")

    def load_hia_from_tdc(self):
        """Load Human Intestinal Absorption from TDC"""
        print("\nLoading HIA (Human Intestinal Absorption)...")
        try:
            data = ADME(name='HIA_Hou')
            df = data.get_data()
            self.datasets['HIA'] = df
            self.task_types['HIA'] = 'classification'
            print(f"✓ HIA: {len(df)} compounds")
            return True
        except Exception as e:
            print(f"✗ HIA failed: {e}")
            return False

    def load_bbb_from_moleculenet(self):
        """Load BBB Penetration from MoleculeNet (DeepChem)"""
        print("\nLoading BBB (Blood-Brain Barrier) from MoleculeNet...")
        try:
            tasks, datasets, transformers = dc.molnet.load_bbbp(
                featurizer='ECFP',
                splitter='scaffold',
                reload=False
            )

            # Combine train, valid, test
            train_dataset, valid_dataset, test_dataset = datasets

            # Extract SMILES and labels
            all_smiles = []
            all_labels = []

            for dataset in [train_dataset, valid_dataset, test_dataset]:
                if hasattr(dataset, 'ids'):
                    all_smiles.extend(dataset.ids)
                    all_labels.extend(dataset.y.flatten())

            df = pd.DataFrame({
                'Drug': all_smiles,
                'Y': all_labels
            })

            self.datasets['BBB'] = df
            self.task_types['BBB'] = 'classification'
            print(f"✓ BBB: {len(df)} compounds")
            return True
        except Exception as e:
            print(f"✗ BBB failed: {e}")
            # Fallback to TDC if available
            try:
                print("  Trying TDC BBB dataset...")
                data = ADME(name='BBB_Martins')
                df = data.get_data()
                self.datasets['BBB'] = df
                self.task_types['BBB'] = 'classification'
                print(f"✓ BBB (TDC): {len(df)} compounds")
                return True
            except:
                return False

    def load_esol_from_moleculenet(self):
        """Load ESOL (Solubility) from MoleculeNet"""
        print("\nLoading ESOL (Aqueous Solubility) from MoleculeNet...")
        try:
            tasks, datasets, transformers = dc.molnet.load_delaney(
                featurizer='ECFP',
                splitter='scaffold',
                reload=False
            )

            # Combine datasets
            train_dataset, valid_dataset, test_dataset = datasets

            all_smiles = []
            all_labels = []

            for dataset in [train_dataset, valid_dataset, test_dataset]:
                if hasattr(dataset, 'ids'):
                    all_smiles.extend(dataset.ids)
                    all_labels.extend(dataset.y.flatten())

            df = pd.DataFrame({
                'Drug': all_smiles,
                'Y': all_labels
            })

            self.datasets['ESOL'] = df
            self.task_types['ESOL'] = 'regression'
            print(f"✓ ESOL: {len(df)} compounds")
            return True
        except Exception as e:
            print(f"✗ ESOL failed: {e}")
            # Fallback to TDC
            try:
                print("  Trying TDC Solubility dataset...")
                data = ADME(name='Solubility_AqSolDB')
                df = data.get_data()
                self.datasets['ESOL'] = df
                self.task_types['ESOL'] = 'regression'
                print(f"✓ Solubility (TDC): {len(df)} compounds")
                return True
            except:
                return False

    def load_lipophilicity_from_tdc(self):
        """Load Lipophilicity (LogP) from TDC"""
        print("\nLoading Lipophilicity (LogP/LogD)...")
        try:
            data = ADME(name='Lipophilicity_AstraZeneca')
            df = data.get_data()
            self.datasets['LogP'] = df
            self.task_types['LogP'] = 'regression'
            print(f"✓ LogP: {len(df)} compounds")
            return True
        except Exception as e:
            print(f"✗ LogP failed: {e}")
            return False

    def load_all_datasets(self):
        """Load all 6 core ADMET datasets"""
        print("="*70)
        print("LOADING 6 CORE ADMET DATASETS")
        print("="*70)

        # Load each dataset
        self.load_herg_from_tdc()
        self.load_cyp_from_tdc()
        self.load_hia_from_tdc()
        self.load_bbb_from_moleculenet()
        self.load_esol_from_moleculenet()
        self.load_lipophilicity_from_tdc()

        # Summary
        print(f"\n{'='*70}")
        print(f"Total datasets loaded: {len(self.datasets)}")
        print(f"Classification tasks: {sum(1 for t in self.task_types.values() if t == 'classification')}")
        print(f"Regression tasks: {sum(1 for t in self.task_types.values() if t == 'regression')}")

        # List loaded tasks
        print(f"\nLoaded tasks:")
        for task, ttype in self.task_types.items():
            count = len(self.datasets[task])
            print(f"  • {task:15s} ({ttype:14s}): {count:6d} compounds")

        print(f"{'='*70}\n")

        return self.datasets, self.task_types
