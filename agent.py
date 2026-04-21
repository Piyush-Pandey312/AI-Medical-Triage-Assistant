# Day 5 — Agent | Piyush Pandey
import joblib, torch, numpy as np, io
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AutoModelForSeq2SeqLM)

BASE = "/content/drive/MyDrive/fdp"

# ── Load all models once at import time ──────────────────────────
_ml_model  = joblib.load(f"{BASE}/ml_model/ml_model.pkl")
_ml_scaler = joblib.load(f"{BASE}/ml_model/ml_scaler.pkl")
_dl_model  = tf.keras.models.load_model(f"{BASE}/dl_model/dl_model.keras")
_dev       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_nlp_tok   = AutoTokenizer.from_pretrained(f"{BASE}/nlp_model")
_nlp_mdl   = AutoModelForSequenceClassification.from_pretrained(
                f"{BASE}/nlp_model").to(_dev)
_nlp_mdl.eval()

_llm_tok   = AutoTokenizer.from_pretrained(f"{BASE}/llm_model")
_llm_mdl   = AutoModelForSeq2SeqLM.from_pretrained(
                f"{BASE}/llm_model").to(_dev)
_llm_mdl.eval()

DL_CLASSES  = {0: "NORMAL", 1: "PNEUMONIA"}
NLP_LABELS  = {0: "CRITICAL", 1: "MODERATE", 2: "MILD"}
SYMPTOM_WORDS = {
    "chest pain","breathlessness","fever","vomiting","headache",
    "dizziness","nausea","weakness","cough","rash","swelling",
    "paralysis","seizure","unconscious","sweating","bleeding",
    "burning","fatigue","palpitations","wheezing","confusion"
}

class TriageAgent:
    def run(self, name, age, bp, chol, sugar, hr, sex,
            complaint, img_bytes=None):
        result = {"name": name, "age": age}

        # Module 1 — ML Risk
        try:
            feat  = [[age, sex, 0, bp, chol, sugar, 0, hr, 0, 1.0, 1, 0, 2]]
            feat  = _ml_scaler.transform(feat)
            score = _ml_model.predict_proba(feat)[0][1]
            label = "HIGH" if score > 0.6 else "MODERATE" if score > 0.35 else "LOW"
            result["risk"] = {"label": label, "score": round(float(score), 2)}
        except Exception as e:
            result["risk"] = {"label": "UNKNOWN", "score": 0.0}

        # Module 2 — DL X-ray (optional)
        if img_bytes:
            try:
                img  = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224,224))
                arr  = preprocess_input(np.array(img, dtype=np.float32))
                arr  = np.expand_dims(arr, 0)
                pred = _dl_model.predict(arr, verbose=0)
                idx  = int(np.argmax(pred))
                result["xray"] = {"label": DL_CLASSES[idx],
                                  "confidence": round(float(np.max(pred)), 2)}
            except Exception:
                result["xray"] = {"label": "Error", "confidence": 0.0}
        else:
            result["xray"] = {"label": "Not provided", "confidence": 0.0}

        # Module 3 — NLP Complaint
        try:
            enc  = _nlp_tok([complaint], padding=True, truncation=True,
                            max_length=128, return_tensors="pt")
            enc  = {k: v.to(_dev) for k, v in enc.items()}
            with torch.no_grad():
                logits = _nlp_mdl(**enc).logits
            pid  = int(torch.argmax(logits))
            syms = [w for w in SYMPTOM_WORDS if w in complaint.lower()]
            result["nlp"] = {"symptoms": list(set(syms)),
                             "urgency": NLP_LABELS[pid]}
        except Exception:
            result["nlp"] = {"symptoms": [], "urgency": "UNKNOWN"}

        # Module 4 — LLM Report
        try:
            syms_str = ", ".join(result["nlp"]["symptoms"]) or "none reported"
            prompt   = (
                f"Patient {name}, age {age}. "
                f"Risk: {result[\'risk\'][\'label\']} ({result[\'risk\'][\'score\']:.0%}). "
                f"X-ray: {result[\'xray\'][\'label\']} ({result[\'xray\'][\'confidence\']:.0%}). "
                f"Symptoms: {syms_str}. Urgency: {result[\'nlp\'][\'urgency\']}. "
                f"Write a 5-sentence clinical summary and recommend action."
            )
            inp = _llm_tok(prompt, return_tensors="pt",
                           truncation=True, max_length=600).to(_dev)
            with torch.no_grad():
                out = _llm_mdl.generate(**inp, max_new_tokens=300,
                                        min_new_tokens=80, num_beams=5,
                                        length_penalty=2.0, no_repeat_ngram_size=3,
                                        repetition_penalty=1.3)
            report = _llm_tok.decode(out[0], skip_special_tokens=True).strip()
            if len(report.split()) < 20:
                raise ValueError("Too short")
            result["report"] = report
        except Exception:
            action = {"CRITICAL": "Immediate emergency care required.",
                      "MODERATE": "Admit for monitoring and treatment.",
                      "MILD":     "Discharge with outpatient follow-up."
                      }.get(result["nlp"]["urgency"], "Clinical evaluation recommended.")
            result["report"] = (
                f"Patient {name}, {age} years, presents with "
                f"{result[\'risk\'][\'label\'].lower()} cardiovascular risk "
                f"and {result[\'xray\'][\'label\'].lower()} on chest imaging. "
                f"Urgency: {result[\'nlp\'][\'urgency\']}. {action}"
            )
        return result
