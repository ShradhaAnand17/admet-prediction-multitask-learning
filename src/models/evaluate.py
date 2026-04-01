# PREDICTION & REPORTING

import torch
import numpy as np
from rdkit import Chem

class ADMETPredictor:
    """Production ADMET predictor"""

    def __init__(self, model, featurizer, task_types, scalers=None):
        self.model = model
        self.featurizer = featurizer
        self.task_types = task_types
        self.scalers = scalers or {}
        self.model.eval()
        self.device = next(model.parameters()).device

    def predict(self, smiles):
        """Predict ADMET properties"""

        # Standardize SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        smiles = Chem.MolToSmiles(mol, canonical=True)

        # Featurize
        features = self.featurizer.featurize(smiles)
        if features is None:
            return None

        # Predict
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(features_tensor)

        # Format results
        results = {'smiles': smiles, 'predictions': {}}

        for task, pred in predictions.items():
            value = pred.cpu().item()

            # Inverse transform if regression
            if task in self.scalers:
                value = self.scalers[task].inverse_transform([[value]])[0][0]

            task_type = self.task_types[task]

            if task_type == 'classification':
                results['predictions'][task] = {
                    'probability': value,
                    'prediction': 'Positive' if value > 0.5 else 'Negative',
                    'confidence': max(value, 1-value),
                    'type': 'classification'
                }
            else:
                results['predictions'][task] = {
                    'value': value,
                    'type': 'regression'
                }

        return results

    def generate_report(self, results):
        """Generate human-readable report"""

        report = []
        report.append("="*70)
        report.append("ADMET PREDICTION REPORT")
        report.append("="*70)
        report.append(f"\nMolecule: {results['smiles']}")

        # Toxicity
        report.append("\n" + "="*70)
        report.append("CARDIOTOXICITY")
        report.append("="*70)

        if 'hERG' in results['predictions']:
            pred = results['predictions']['hERG']
            prob = pred['probability']
            risk = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"
            report.append(f"\nhERG Inhibition: {risk} RISK")
            report.append(f"  Probability: {prob*100:.1f}%")
            report.append(f"  Prediction: {pred['prediction']}")
            report.append(f"  Confidence: {pred['confidence']*100:.1f}%")

        # CYP Inhibition
        report.append("\n" + "="*70)
        report.append("DRUG-DRUG INTERACTION (CYP450 Inhibition)")
        report.append("="*70)

        cyp_tasks = ['CYP1A2', 'CYP2C9', 'CYP2C19', 'CYP2D6', 'CYP3A4']
        for task in cyp_tasks:
            if task in results['predictions']:
                pred = results['predictions'][task]
                prob = pred['probability']
                status = "INHIBITOR" if prob > 0.5 else "NON-INHIBITOR"
                report.append(f"\n{task}: {status}")
                report.append(f"  Probability: {prob*100:.1f}%")

        # ADME
        report.append("\n" + "="*70)
        report.append("ADME PROPERTIES")
        report.append("="*70)

        if 'HIA' in results['predictions']:
            pred = results['predictions']['HIA']
            report.append(f"\nHuman Intestinal Absorption: {pred['prediction']}")
            report.append(f"  Probability: {pred['probability']*100:.1f}%")

        if 'BBB' in results['predictions']:
            pred = results['predictions']['BBB']
            report.append(f"\nBlood-Brain Barrier: {pred['prediction']}")
            report.append(f"  Probability: {pred['probability']*100:.1f}%")

        # Physicochemical
        report.append("\n" + "="*70)
        report.append("PHYSICOCHEMICAL PROPERTIES")
        report.append("="*70)

        if 'ESOL' in results['predictions']:
            pred = results['predictions']['ESOL']
            value = pred['value']
            report.append(f"\nSolubility (ESOL): {value:.3f} log(mol/L)")
            solubility_class = "Highly soluble" if value > -1 else "Moderately soluble" if value > -3 else "Poorly soluble"
            report.append(f"  Classification: {solubility_class}")

        if 'LogP' in results['predictions']:
            pred = results['predictions']['LogP']
            value = pred['value']
            report.append(f"\nLipophilicity (LogP): {value:.3f}")
            lipophilicity = "Hydrophilic" if value < 2 else "Balanced" if value < 4 else "Lipophilic"
            report.append(f"  Classification: {lipophilicity}")

        # Summary
        report.append("\n" + "="*70)
        report.append("SUMMARY & RECOMMENDATIONS")
        report.append("="*70)

        issues = []

        if 'hERG' in results['predictions']:
            if results['predictions']['hERG']['probability'] > 0.7:
                issues.append("⚠️  HIGH cardiotoxicity risk - structural modification recommended")

        cyp_inhibitors = sum(1 for task in cyp_tasks
                            if task in results['predictions']
                            and results['predictions'][task]['probability'] > 0.6)
        if cyp_inhibitors >= 3:
            issues.append("⚠️  Multiple CYP inhibition - significant DDI risk")

        if 'HIA' in results['predictions']:
            if results['predictions']['HIA']['probability'] < 0.3:
                issues.append("⚠️  Poor oral absorption - alternative delivery may be needed")

        if issues:
            report.append("\nCritical Issues:")
            for issue in issues:
                report.append(f"  {issue}")
        else:
            report.append("\n✓ No critical safety concerns identified")

        report.append("\n" + "="*70)

        return "\n".join(report)
