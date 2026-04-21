
# Day 3 — NLP Module
import torch
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = '/content/drive/MyDrive/fdp/nlp_model'
LABEL_MAP  = {0: 'CRITICAL', 1: 'MODERATE', 2: 'MILD'}
SYMPTOM_WORDS = {
    'chest pain', 'breathlessness', 'fever', 'vomiting', 'headache',
    'dizziness', 'nausea', 'weakness', 'cough', 'rash', 'swelling',
    'paralysis', 'seizure', 'unconscious', 'sweating', 'bleeding',
    'burning', 'fatigue', 'palpitations', 'wheezing', 'confusion'
}

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

nlp_spacy = spacy.load('en_core_web_sm')

def extract_symptoms(text):
    return list({w for w in SYMPTOM_WORDS if w in text.lower()})

def analyze_complaint(text):
    enc = tokenizer([text], padding=True, truncation=True,
                    max_length=128, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    pred_id  = int(torch.argmax(logits))
    return {
        'symptoms': extract_symptoms(text),
        'urgency' : LABEL_MAP[pred_id]
    }
