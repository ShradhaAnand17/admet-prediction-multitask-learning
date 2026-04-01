"""
Web Deployment - INDUSTRY-READY ADMET PREDICTOR  
"""
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import pickle, os, io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from pyngrok import ngrok
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ============================================================================
# RISK LOGIC (Traffic Light System)

def get_risk_assessment(task, pred_data):
    """
    Unified risk assessment function for both web UI and PDF generation.
    Returns: (label, background_color, text_color)
    """
    if pred_data['type'] == 'classification':
        prob = pred_data['probability']

        if task == 'hERG':
            if prob > 0.7:
                return 'High Risk', '#fee2e2', '#ef4444'
            if prob > 0.4:
                return 'Medium Risk', '#fef3c7', '#f59e0b'
            return 'Safe', '#d1fae5', '#10b981'

        elif task.startswith('CYP'):
            if prob > 0.6:
                return 'Inhibitor', '#fee2e2', '#ef4444'
            return 'Safe', '#d1fae5', '#10b981'

        else:  # BBB, HIA
            if prob > 0.6:
                return 'Safe', '#d1fae5', '#10b981'
            if prob > 0.3:
                return 'Moderate', '#fef3c7', '#f59e0b'
            return 'Risk', '#fee2e2', '#ef4444'

    else:  # Regression (ESOL, LogP)
        val = pred_data['value']
        if task == 'ESOL':
            if val > -1:
                return 'High Solubility', '#d1fae5', '#10b981'
            if val > -3:
                return 'Moderate Solubility', '#fef3c7', '#f59e0b'
            return 'Poor Solubility', '#fee2e2', '#ef4444'
        elif task == 'LogP':
            if 2 <= val <= 4:
                return 'Optimal', '#d1fae5', '#10b981'
            if val < 2 or val > 4:
                return 'Suboptimal', '#fef3c7', '#f59e0b'
            return 'Poor', '#fee2e2', '#ef4444'

    return 'Neutral', '#e5e7eb', '#6b7280'

# ============================================================================
# MODEL CLASSES (Imported Logic)
# [These classes must match your training logic exactly]

class MolecularFeaturizer:
    def __init__(self, radius=2, n_bits=2048, use_descriptors=True):
        self.radius, self.n_bits, self.use_descriptors = radius, n_bits, use_descriptors
    def featurize(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        from rdkit.Chem import AllChem, Descriptors
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits))
        if self.use_descriptors:
            desc = [Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
                    Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
                    Descriptors.NumRotatableBonds(mol), Descriptors.NumAromaticRings(mol),
                    Descriptors.RingCount(mol), Descriptors.HeavyAtomCount(mol)]
            return np.concatenate([fp, np.array(desc)])
        return fp
    def get_feature_dim(self): return self.n_bits + (9 if self.use_descriptors else 0)

class MultiTaskADMETModel(torch.nn.Module):
    def __init__(self, input_dim, task_names, task_types, hidden_dims=[1024, 512, 256]): # Added hidden_dims
        super().__init__()
        self.task_names = task_names

        # Shared Backbone - Reconstructed to match training logic
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                torch.nn.Linear(prev_dim, h),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(h),
                torch.nn.Dropout(0.3)
            ])
            prev_dim = h
        self.backbone = torch.nn.Sequential(*layers)

        self.heads = torch.nn.ModuleDict({
            t: torch.nn.Sequential(
                torch.nn.Linear(hidden_dims[-1], 128), # Input to head uses last hidden_dim
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1)
            ) for t in task_names
        })
    def forward(self, x):
        shared = self.backbone(x)
        return {t: self.heads[t](shared) for t in self.task_names}

class ADMETPredictor:
    def __init__(self, model, featurizer, task_types, scalers=None):
        self.model, self.featurizer, self.task_types, self.scalers = model, featurizer, task_types, scalers or {}
        self.device = next(model.parameters()).device
    def predict(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        smiles = Chem.MolToSmiles(mol, canonical=True)
        features = self.featurizer.featurize(smiles)
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        with torch.no_grad(): preds = self.model(x)
        results = {'smiles': smiles, 'predictions': {}}
        for t, p in preds.items():
            val = p.cpu().item()
            if self.task_types[t] == 'classification': val = 1 / (1 + np.exp(-val))
            if t in self.scalers: val = self.scalers[t].inverse_transform([[val]])[0][0]
            results['predictions'][t] = {'probability': val if self.task_types[t]=='classification' else None,
                                        'value': val if self.task_types[t]=='regression' else None,
                                        'prediction': ('Positive' if val > 0.5 else 'Negative') if self.task_types[t]=='classification' else 'Value',
                                        'type': self.task_types[t]}
        return results

# ============================================================================
# FLASK INTERFACE

predictor = None

def load_admet_system():
    global predictor
    task_types = {'hERG': 'classification', 'CYP1A2': 'classification', 'CYP2C9': 'classification', 'CYP2C19': 'classification', 'CYP2D6': 'classification', 'CYP3A4': 'classification', 'HIA': 'classification', 'BBB': 'classification', 'ESOL': 'regression', 'LogP': 'regression'}
    feat = MolecularFeaturizer()
    # Pass hidden_dims explicitly to match training model
    model = MultiTaskADMETModel(feat.get_feature_dim(), list(task_types.keys()), task_types, hidden_dims=[1024, 512, 256])
    if os.path.exists('best_admet_model.pt'):
        model.load_state_dict(torch.load('best_admet_model.pt', map_location='cpu', weights_only=False)['model_state'])
        model.eval()
        sc = pickle.load(open('regression_scalers.pkl', 'rb')) if os.path.exists('regression_scalers.pkl') else {}
        predictor = ADMETPredictor(model, feat, task_types, sc)
        return True
    return False

@app.route('/', methods=['GET'])
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>ADMET AI Predictor</title>
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700;900&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            :root { --primary: #1a5f7a; --accent: #159895; --text: #2c3333; }
            body { font-family: 'Inter', sans-serif; background: #f8f9fa; margin: 0; padding: 40px; color: var(--text); }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 20px; box-shadow: 0 4px 30px rgba(0,0,0,0.05); }

            /* Header & Examples */
            h1 { color: var(--primary); text-align: center; font-size: 2.8em; font-weight: 900; margin-bottom: 20px; }
            .examples { text-align: center; margin-bottom: 30px; }
            .ex-btn { padding: 8px 18px; margin: 5px; border: 2px solid var(--primary); background: white; border-radius: 25px; cursor: pointer; color: var(--primary); font-weight: 600; transition: 0.3s; }
            .ex-btn:hover { background: var(--primary); color: white; }

            /* Search Bar Section */
            .search-area { display: flex; gap: 15px; margin-bottom: 40px; }
            input { flex: 1; padding: 18px; border: 2px solid #e5e7eb; border-radius: 12px; font-family: monospace; font-size: 1.1em; }
            .btn { padding: 15px 35px; border-radius: 12px; border: none; font-weight: 700; cursor: pointer; color: white; font-size: 1.1em; }
            .btn-p { background: var(--primary); }
            .btn-s { background: var(--accent); }

            /* Key Insights Grid (Cards) */
            .insight-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 40px; }
            .insight-card { background: var(--accent); padding: 25px; border-radius: 15px; color: white; }
            .insight-card h3 { font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 10px 0; opacity: 0.9; }
            .insight-val { font-size: 2.5em; font-weight: 900; font-family: 'Roboto'; }

            /* Professional Data Table */
            table { width: 100%; border-collapse: collapse; margin-top: 20px; border-radius: 12px; overflow: hidden; }
            th { background: var(--primary); color: white; padding: 18px; text-align: left; text-transform: uppercase; font-size: 0.85em; }
            td { padding: 15px 18px; border-bottom: 1px solid #f1f5f9; font-size: 0.95em; }
            tr:hover { background: #f0f9ff; }
            .badge { padding: 6px 15px; border-radius: 25px; font-size: 12px; font-weight: 700; text-transform: capitalize; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1> ADMET AI Predictor</h1>

            <div class="examples">
                <strong>EXAMPLES:</strong>
                <button class="ex-btn" onclick="setS('CC(=O)Oc1ccccc1C(=O)O')">Aspirin</button>
                <button class="ex-btn" onclick="setS('CC(C)Cc1ccc(cc1)C(C)C(=O)O')">Ibuprofen</button>
                <button class="ex-btn" onclick="setS('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')">Caffeine</button>
                <button class="ex-btn" onclick="setS('CC(=O)Nc1ccc(O)cc1')">Paracetamol</button>
            </div>

            <div class="search-area">
                <input type="text" id="smiles" placeholder="Enter Molecule SMILES string...">
                <button class="btn btn-p" onclick="predict()">Analyze</button>
                <button class="btn btn-s" id="dl" onclick="download()" disabled>Download PDF</button>
            </div>

            <div id="insights" class="insight-grid" style="display:none">
                <div class="insight-card"><h3>Toxicity (hERG)</h3><div id="i-herg" class="insight-val">--</div></div>
                <div class="insight-card"><h3>Absorption (HIA)</h3><div id="i-hia" class="insight-val">--</div></div>
                <div class="insight-card"><h3>Solubility</h3><div id="i-sol" class="insight-val">--</div></div>
                <div class="insight-card"><h3>DDI Risk</h3><div id="i-ddi" class="insight-val">--</div></div>
            </div>

            <div id="res" style="display:none">
                <table>
                    <thead><tr><th>Property</th><th>Prediction</th><th>Probability/Value</th><th>Status</th></tr></thead>
                    <tbody id="tbody"></tbody>
                </table>
            </div>
        </div>

        <script>
            function setS(s) { document.getElementById('smiles').value = s; }

            async function predict() {
                const s = document.getElementById('smiles').value;
                const r = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({smiles:s})});
                const d = await r.json();
                if(d.success) {
                    document.getElementById('insights').style.display = 'grid';
                    document.getElementById('res').style.display = 'block';
                    document.getElementById('dl').disabled = false;

                    // Update Insight Cards
                    document.getElementById('i-herg').innerText = (d.predictions.hERG.probability*100).toFixed(0)+'%';
                    document.getElementById('i-hia').innerText = (d.predictions.HIA.probability*100).toFixed(0)+'%';
                    document.getElementById('i-sol').innerText = d.predictions.ESOL.value.toFixed(1);
                    const inhibs = ['CYP1A2','CYP2C9','CYP2C19','CYP2D6','CYP3A4'].filter(c=>d.predictions[c].probability > 0.6).length;
                    document.getElementById('i-ddi').innerText = inhibs + '/5';

                    // Build Table with Traffic Lights
                    let h = '';
                    for(const [task, p] of Object.entries(d.predictions)) {
                        const risk = d.risk_info[task];
                        h += `<tr>
                            <td><b>${task}</b></td>
                            <td>${p.prediction}</td>
                            <td>${p.type=='classification'?(p.probability*100).toFixed(1)+'%':p.value.toFixed(3)}</td>
                            <td><span class="badge" style="background:${risk.bg}; color:${risk.text}">${risk.label}</span></td>
                        </tr>`;
                    }
                    document.getElementById('tbody').innerHTML = h;
                }
            }
            async function download() {
                const s = document.getElementById('smiles').value;
                const r = await fetch('/download_report', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({smiles:s})});
                const b = await r.blob(); const u = window.URL.createObjectURL(b);
                const a = document.createElement('a'); a.href = u; a.download = 'ADMET_Report.pdf'; a.click();
            }
        </script>
    </body>
    </html>
    """

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        results = predictor.predict(data.get('smiles'))
        if not results:
            return jsonify({'success': False, 'error': 'Invalid SMILES'}), 400

        # Calculate risk info for frontend badges
        risk_info = {}
        for t, p in results['predictions'].items():
            label, bg, text = get_risk_assessment(t, p)  # ✅ CORRECT FUNCTION NAME
            risk_info[t] = {"label": label, "bg": bg, "text": text}

        return jsonify({
            'success': True,
            'smiles': results['smiles'],
            'predictions': results['predictions'],
            'risk_info': risk_info
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/download_report', methods=['POST'])
def download_report():
    try:
        data = request.get_json()
        smiles = data.get('smiles')
        results = predictor.predict(smiles)

        pdf_buffer = io.BytesIO()
        # Adjusted margins to ensure a single-page fit
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4,
                               topMargin=0.4*inch, bottomMargin=0.4*inch,
                               leftMargin=0.5*inch, rightMargin=0.5*inch)
        elements = []
        styles = getSampleStyleSheet()

        # --- Professional Custom Styles ---
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=20,
                                   textColor=colors.HexColor('#1a5f7a'), alignment=TA_CENTER, spaceAfter=5)
        date_style = ParagraphStyle('Date', alignment=TA_CENTER, fontSize=9, textColor=colors.grey, spaceAfter=20)
        heading_style = ParagraphStyle('Heading', fontSize=12, textColor=colors.HexColor('#1a5f7a'),
                                     fontName='Helvetica-Bold', spaceBefore=15, spaceAfter=10)
        normal_style = styles['Normal']

        # 1. Main Title & Date
        elements.append(Paragraph("ADMET PROPERTY PREDICTION REPORT", title_style))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", date_style))

        # 2. Molecular Structure Image (Strategic Positioning)
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                img = Draw.MolToImage(mol, size=(300, 300))
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                mol_img = RLImage(img_buffer, width=1.6*inch, height=1.6*inch)
                mol_img.hAlign = 'CENTER'
                elements.append(mol_img)
                elements.append(Spacer(1, 0.2*inch))
        except Exception as e:
            print(f"Image generation failed: {e}")

        # 3. Drug Identification Section
        elements.append(Paragraph("DRUG IDENTIFICATION", heading_style))
        smiles_style = ParagraphStyle('SMILES', parent=normal_style, fontSize=9, fontName='Courier',
                                    leftIndent=20, leading=11, textColor=colors.black)
        elements.append(Paragraph(f"<b>SMILES ID:</b> {smiles}", smiles_style))

        # 4. Executive Summary Section (Spaced & Bulleted)
        elements.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
        preds = results['predictions']
        summary_points = []

        # Logic-driven summary sentences
        h_risk, _, _ = get_risk_assessment('hERG', preds['hERG'])
        summary_points.append(f"<b>Toxicity (hERG):</b> Predicted as <b>{h_risk}</b> ({preds['hERG']['probability']*100:.1f}% probability).")

        cyps = ['CYP1A2','CYP2C9','CYP2C19','CYP2D6','CYP3A4']
        inhibs = [c for c in cyps if preds[c]['probability'] > 0.6]
        summary_points.append(f"<b>Metabolism:</b> " + (f"Potential inhibitor of <b>{', '.join(inhibs)}</b>." if inhibs else "No significant CYP inhibition detected. Low risk of drug-drug interactions."))

        summary_points.append(f"<b>Absorption/Distribution:</b> Intestinal Absorption (HIA) is <b>{preds['HIA']['prediction']}</b>; BBB Penetration is <b>{preds['BBB']['prediction']}</b>.")

        if 'ESOL' in preds:
            sol_status, _, _ = get_risk_assessment('ESOL', preds['ESOL'])
            summary_points.append(f"<b>Solubility:</b> Predicted aqueous solubility is <b>{sol_status}</b> ({preds['ESOL']['value']:.2f} log mol/L).")

        bullet_style = ParagraphStyle('Bullet', parent=normal_style, fontSize=10, leftIndent=35,
                                    firstLineIndent=-15, spaceBefore=4, leading=14)
        for point in summary_points:
            elements.append(Paragraph(f"• {point}", bullet_style))

        # 5. Detailed Prediction Data Table (Clean Industrial Look)
        elements.append(Paragraph("DETAILED PREDICTION DATA", heading_style))
        table_data = [['Property', 'Prediction', 'Confidence/Value', 'Risk Status']]

        for t, p in preds.items():
            r_label, r_hex, _ = get_risk_assessment(t, p)
            val = f"{p['probability']*100:.1f}%" if p['type']=='classification' else f"{p['value']:.3f}"
            table_data.append([t, p['prediction'], val, r_label])

        # Adjusted column widths for balance
        table = Table(table_data, colWidths=[1.4*inch, 1.4*inch, 1.4*inch, 1.4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5f7a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(table)

        # 6. Footer Disclaimer
        elements.append(Spacer(1, 0.3*inch))
        elements.append(Paragraph("<i>Note: AI-generated report for research use only. Results should be validated experimentally.</i>",
                                 ParagraphStyle('Disclaimer', alignment=TA_CENTER, fontSize=7, textColor=colors.grey)))

        # Build PDF
        doc.build(elements)
        pdf_buffer.seek(0)

        return Response(pdf_buffer.read(), mimetype='application/pdf',
                        headers={'Content-Disposition': 'attachment; filename=ADMET_Report.pdf'})

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if load_admet_system():
        import os
        ngrok.set_auth_token("put your ngrok auth token here")
        url = ngrok.connect(5000).public_url
        print(f"\n🚀 SUCCESS! TOOL RUNNING AT: {url}\n")
        app.run(port=5000)
