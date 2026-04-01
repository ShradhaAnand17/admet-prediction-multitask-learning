class MolecularFeaturizer:
    """Generate molecular fingerprints and descriptors"""

    def __init__(self, radius=2, n_bits=2048, use_descriptors=True):
        self.radius = radius
        self.n_bits = n_bits
        self.use_descriptors = use_descriptors

    def smiles_to_mol(self, smiles):
        """Convert SMILES to RDKit mol object with standardization"""
        # Removed the try-except block, as Chem.MolFromSmiles returns None for invalid SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return mol

    def get_morgan_fingerprint(self, smiles):
        """Generate Morgan (ECFP) fingerprint"""
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            return None

        # Removed try-except to expose RDKit errors
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, self.radius, nBits=self.n_bits
        )
        return np.array(fp)

    def calculate_descriptors(self, smiles):
        """Calculate key physicochemical descriptors"""
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            return None

        # Removed try-except to expose RDKit errors
        descriptors = [
            Descriptors.MolWt(mol),                     # Molecular weight
            Descriptors.MolLogP(mol),                   # LogP
            Descriptors.TPSA(mol),                      # Polar surface area
            Descriptors.NumHDonors(mol),                # H-bond donors
            Descriptors.NumHAcceptors(mol),             # H-bond acceptors
            Descriptors.NumRotatableBonds(mol),         # Rotatable bonds
            Descriptors.NumAromaticRings(mol),          # Aromatic rings
            Descriptors.RingCount(mol),                 # Total rings
            # Descriptors.CalcFractionCSP3(mol),        # Fraction sp3 carbon - temporarily removed due to persistent AttributeError
            Descriptors.HeavyAtomCount(mol),            # Heavy atoms
        ]
        return np.array(descriptors)

    def featurize(self, smiles):
        """Generate complete feature vector"""
        fp = self.get_morgan_fingerprint(smiles)
        if fp is None:
            return None

        if self.use_descriptors:
            desc = self.calculate_descriptors(smiles)
            if desc is None:
                return None
            features = np.concatenate([fp, desc])
        else:
            features = fp

        return features

    def get_feature_dim(self):
        """Get total feature dimension"""
        # Adjusted for the removed descriptor
        return self.n_bits + (9 if self.use_descriptors else 0)
