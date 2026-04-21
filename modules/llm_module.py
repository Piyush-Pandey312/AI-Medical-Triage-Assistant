# Day 4 — LLM Module
# Loads Flan-T5 from saved Drive path (faster than downloading each time)
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "/content/drive/MyDrive/fdp/llm_model"   # local Drive path
_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading Flan-T5 from Drive on {_dev}...")
_tok = AutoTokenizer.from_pretrained(MODEL_PATH)
_mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(_dev)
_mdl.eval()
print("✅ Flan-T5 loaded from Drive")

def _build_prompt(name, age, risk, xray, nlp_result):
    syms = ", ".join(nlp_result["symptoms"]) if nlp_result["symptoms"] else "none reported"
    urgency_desc = {
        "CRITICAL": "life-threatening, requires immediate intervention",
        "MODERATE": "serious but stable, requires prompt attention",
        "MILD":     "non-urgent, can be managed with routine care"
    }.get(nlp_result["urgency"], nlp_result["urgency"])
    return (
        f"You are a senior emergency medicine physician writing a formal clinical triage report. "
        f"A patient named {name}, {age} years old, has been brought to the emergency department. "
        f"The AI-based risk assessment has classified the cardiovascular risk as {risk[\'label\']} "
        f"with a probability score of {risk[\'score\']:.0%}. "
        f"The chest X-ray analysis returned a finding of {xray[\'label\']} "
        f"with a confidence of {xray[\'confidence\']:.0%}. "
        f"The patient reported the following symptoms: {syms}. "
        f"The overall triage urgency is {nlp_result[\'urgency\']}, meaning {urgency_desc}. "
        f"Write a detailed 5-sentence medical summary covering: "
        f"1) Patient and complaints. 2) AI risk and X-ray findings. "
        f"3) Clinical interpretation. 4) Urgency meaning. 5) Recommended action."
    )

def _rule_based(name, age, risk, xray, nlp_result):
    syms = ", ".join(nlp_result["symptoms"]) if nlp_result["symptoms"] else "none reported"
    action = {
        "CRITICAL": (
            "This patient must be transferred to the emergency resuscitation bay immediately. "
            "IV access, cardiac monitoring, and oxygen supplementation should be initiated without delay. "
            "A senior physician and crash team must be alerted at once."
        ),
        "MODERATE": (
            "The patient should be admitted to a monitored ward for further investigation. "
            "Blood investigations, ECG, and specialist consultation are recommended. "
            "Regular vitals monitoring every 2 hours is advised until stable."
        ),
        "MILD": (
            "The patient may be discharged with appropriate medications and advice. "
            "A follow-up with a general physician within 48 hours is recommended. "
            "Patient should return immediately if symptoms worsen."
        )
    }.get(nlp_result["urgency"], "Clinical evaluation by a physician is recommended.")
    risk_explain = {
        "HIGH":     "indicating significant likelihood of underlying cardiac pathology",
        "MODERATE": "suggesting moderate probability of cardiovascular involvement",
        "LOW":      "suggesting low probability of acute cardiac disease"
    }.get(risk["label"], "")
    xray_explain = {
        "PNEUMONIA": "consistent with active pulmonary infection requiring treatment",
        "NORMAL":    "showing no acute cardiopulmonary abnormality at this time"
    }.get(xray["label"], "")
    return (
        f"Patient {name}, {age} years of age, has presented to the emergency department "
        f"with the following reported symptoms: {syms}. "
        f"The AI risk assessment classified cardiovascular risk as {risk[\'label\']} "
        f"({risk[\'score\']:.0%} probability), {risk_explain}. "
        f"Chest X-ray returned {xray[\'label\']} ({xray[\'confidence\']:.0%} confidence), {xray_explain}. "
        f"Overall triage urgency is {nlp_result[\'urgency\']}, warranting timely clinical response. "
        f"{action}"
    )

def generate_report(name, age, risk, xray, nlp_result):
    prompt = _build_prompt(name, age, risk, xray, nlp_result)
    try:
        inp = _tok(prompt, return_tensors="pt", truncation=True, max_length=600).to(_dev)
        with torch.no_grad():
            out = _mdl.generate(
                **inp,
                max_new_tokens=300,
                min_new_tokens=80,
                num_beams=5,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3
            )
        report = _tok.decode(out[0], skip_special_tokens=True).strip()
        if len(report.split()) < 30:
            raise ValueError("Output too short, using fallback")
        return {"report": report}
    except Exception as e:
        print(f"Flan fallback triggered: {e}")
        return {"report": _rule_based(name, age, risk, xray, nlp_result)}
