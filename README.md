# 🏥 AI-Powered Medical Triage Assistant
> **AI Project — Built end to end by Piyush Pandey**
> HuggingFace + Google Colab T4 GPU + Gradio + Gmail Agent

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square)
![Gradio](https://img.shields.io/badge/Gradio-Live_UI-green?style=flat-square)
![NIELIT](https://img.shields.io/badge/NIELIT-AI%2FML%20Intern-blueviolet?style=flat-square)

---

## 👨‍💻 About the Developer

| | |
|---|---|
| **Name** | Piyush Pandey |
| **Role** | AI/ML Intern |
| **Organization** | NIELIT (National Institute of Electronics & Information Technology) |
| **Project Type** | End-to-End AI Project — Designed, Built & Deployed Independently |

---

## 📌 Project Overview

A real clinical decision-support system that accepts patient vitals,
a chest X-ray image, and a free-text symptom complaint. It predicts
disease risk (ML), classifies the X-ray (DL), extracts urgency from
text (NLP), generates a doctor-readable summary (LLM), and
automatically routes a referral email to the relevant specialist —
all displayed on a live Gradio dashboard running on Google Colab.

### 🎯 What Makes This Special
- **Every AI domain covered** — ML, DL, NLP, LLM, Agent in one unified project
- **Zero local setup** — runs entirely on free Google Colab T4 GPU
- **Zero cost datasets** — all loaded from HuggingFace Hub in one line of code
- **Live demo URL** — shareable Gradio link works on any phone or browser
- **Real-world usefulness** — mirrors AI triage systems used in Apollo, AIIMS

---
## 🏗️ System Architecture

```
Patient Input (Gradio UI)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│                   TriageAgent                        │
│                                                      │
│  Step 1 ──► predict_risk()        [ML Module]       │
│             Random Forest                            │
│             → { label: HIGH, score: 0.78 }          │
│                                                      │
│  Step 2 ──► classify_xray()       [DL Module]       │
│             MobileNetV2                              │
│             → { label: PNEUMONIA, conf: 0.91 }      │
│                                                      │
│  Step 3 ──► analyze_complaint()   [NLP Module]      │
│             DistilBERT Fine-tuned                    │
│             → { symptoms: [...], urgency: CRITICAL } │
│                                                      │
│  Step 4 ──► generate_report()     [LLM Module]      │
│             Flan-T5-large                            │
│             → { report: "5-sentence summary..." }   │
│                                                      │
│  Step 5 ──► send_referral_email() [Agent]           │
│             Gmail SMTP → Specialist Doctor           │
└─────────────────────────────────────────────────────┘
         │
         ▼
  Gradio Dashboard (Live Public URL)
```

## 📁 Project Structure

```
fdp-medical-triage/
│
├── 📓 notebooks/
│   ├── day1_ml_colab.ipynb          # ML — Heart disease risk
│   ├── day2_dl_colab.ipynb          # DL — Chest X-ray classification
│   ├── day3_nlp_colab.ipynb         # NLP — Urgency classification
│   ├── day4_llm_colab.ipynb         # LLM — Clinical report generation
│   └── day5_agent_colab.ipynb       # Agent + Gradio + Email
│
├── 📄 modules/
│   ├── ml_module.py                 # predict_risk()
│   ├── dl_module.py                 # classify_xray()
│   ├── nlp_module.py                # analyze_complaint()
│   └── llm_module.py                # generate_report()
│
├── 🤖 agent.py                      # TriageAgent orchestrator
├── 🖥️  app.py                        # Gradio web UI + Email agent
├── 📋 requirements.txt              # All dependencies
├── 🔒 .gitignore                    # Excludes secrets and models
└── 📖 README.md                     # This file
```
---

## 🤖 AI Modules Built

### Module 1 — Machine Learning
| | |
|---|---|
| **Task** | Heart disease risk prediction |
| **Dataset** | `aai530-group6/heart-failure-prediction-dataset` from HuggingFace |
| **Model** | Random Forest Classifier |
| **Accuracy** | 82–88% |
| **Output** | `{ "label": "HIGH", "score": 0.78 }` |

### Module 2 — Deep Learning
| | |
|---|---|
| **Task** | Chest X-ray classification |
| **Dataset** | `keremberke/chest-xray-classification` from HuggingFace |
| **Model** | MobileNetV2 (Transfer Learning) |
| **Accuracy** | 88–94% |
| **Output** | `{ "label": "PNEUMONIA", "confidence": 0.91 }` |

### Module 3 — NLP
| | |
|---|---|
| **Task** | Patient complaint urgency classification |
| **Dataset** | Custom 60-sample synthetic medical complaints |
| **Model** | DistilBERT fine-tuned (3-class) |
| **Accuracy** | 85–92% |
| **Output** | `{ "symptoms": ["chest pain"], "urgency": "CRITICAL" }` |

### Module 4 — LLM
| | |
|---|---|
| **Task** | Clinical report generation |
| **Model** | google/flan-t5-large (local GPU) |
| **Technique** | Prompt engineering + beam search |
| **Output** | `{ "report": "5-sentence clinical summary..." }` |

### Module 5 — AI Agent
| | |
|---|---|
| **Task** | Orchestration + UI + Email routing |
| **Framework** | Gradio + Gmail SMTP |
| **Routing** | Auto-assigns specialist based on AI findings |
| **Output** | Live public URL + referral email to doctor |

---

## 🔄 Doctor Referral Routing Logic

| Finding | Urgency | Assigned Specialist |
|---------|---------|---------------------|
| HIGH cardiovascular risk | CRITICAL | 🔴 Cardiologist + CC Emergency |
| PNEUMONIA on X-ray | CRITICAL | 🔴 Pulmonologist + CC Emergency |
| PNEUMONIA on X-ray | MODERATE | 🟡 Pulmonologist |
| HIGH / MODERATE risk | MODERATE | 🟡 Cardiologist |
| Neurological symptoms | MODERATE | 🟡 Neurologist |
| Any | MILD | 🟢 General Physician |

---

## 🚀 How to Run

### Step 1 — Open in Google Colab
Runtime → Change runtime type → T4 GPU → Save

### Step 2 — Install dependencies
```bash
pip install transformers datasets torch tensorflow
pip install gradio scikit-learn joblib pillow spacy accelerate
python -m spacy download en_core_web_sm
```

### Step 3 — Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 4 — Run notebooks in order
day1 → day2 → day3 → day4 → day5
Each notebook saves its model to Google Drive automatically.

### Step 5 — Launch Gradio App
```python
demo.launch(share=True)
# Prints a public URL → open on any phone or browser
```

---

## 📧 Email Agent Setup

### Gmail App Password

myaccount.google.com
Security → 2-Step Verification → Turn ON
Security → App Passwords → Generate
Paste 16-character password in SENDER_PASSWORD


### Configure credentials
```python
SENDER_EMAIL    = 'emergency@yourhospital.com'
SENDER_PASSWORD = 'xxxx xxxx xxxx xxxx'
```

---

## 📊 Model Storage (Google Drive)

```
Models are saved to Drive and loaded at runtime.
Not committed to this repo due to file size.
Google Drive/fdp/
├── 📂 ml_model/
│   ├── ml_model.pkl       (2.4 MB)
│   └── ml_scaler.pkl      (1 KB)
├── 📂 dl_model/
│   ├── dl_model.keras     (22.7 MB)
│   └── dl_model.h5        (22.5 MB)
├── 📂 nlp_model/
│   └── model.safetensors  (255.4 MB)
└── 📂 llm_model/
└── model.safetensors  (2.92 GB)
```

---

## 📈 Results Summary

| Module | Model | Metric |
|--------|-------|--------|
| ML — Risk Prediction | Random Forest | 82–88% Accuracy |
| DL — X-Ray Classification | MobileNetV2 | 88–94% Accuracy |
| NLP — Urgency Detection | DistilBERT | 85–92% Accuracy |
| LLM — Report Generation | Flan-T5-large | 5-sentence clinical output |
| Agent — Email Routing | Gmail SMTP | Auto specialist assignment |

---

## 📦 Requirements

```txt
torch
transformers
tensorflow
scikit-learn
joblib
pillow
spacy
numpy
gradio
accelerate
secure-smtplib
datasets
```

---

## 🔒 .gitignore
Model files
*.pkl
*.h5
*.keras
*.safetensors
*.bin
Secrets
secrets.py
.env
Python
pycache/
*.pyc
.ipynb_checkpoints/
Drive
drive/

---

## 🌐 Live Demo
Open day5_agent_colab.ipynb in Google Colab
→ Run all cells
→ Gradio prints a public URL
→ Share with anyone — works on phone, tablet, browser
→ URL stays live for 72 hours

---

## 🙏 Acknowledgements

- [HuggingFace](https://huggingface.co) — datasets and pretrained models
- [Google Colab](https://colab.research.google.com) — free T4 GPU
- [Gradio](https://gradio.app) — live demo interface
- [NIELIT](https://nielit.gov.in) — internship and guidance

---

<p align="center">
  <b>Built with ❤️ by Piyush Pandey</b><br>
  AI/ML Intern — NIELIT<br>
  <i>AI-Powered Medical Triage Assistant</i>
</p>
