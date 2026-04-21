# Day 5 — Streamlit App | Piyush Pandey
import streamlit as st
import io
from agent import TriageAgent

st.set_page_config(
    page_title="AI Medical Triage Assistant",
    page_icon="🏥",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────
st.title("🏥 AI-Powered Medical Triage Assistant")
st.caption("FDP Demo — 5-Day AI Project │ ML + DL + NLP + LLM │ HuggingFace + Colab")
st.divider()

# ── Sidebar: Patient Input ────────────────────────────────────────
with st.sidebar:
    st.header("🧑‍⚕️ Patient Details")
    name    = st.text_input("Full Name", "Ramesh Kumar")
    age     = st.slider("Age", 18, 90, 55)
    sex     = st.radio("Sex", ["Male", "Female"])
    sex_int = 1 if sex == "Male" else 0
    st.divider()
    st.subheader("Vitals")
    bp      = st.slider("Blood Pressure (mmHg)", 80, 200, 160)
    chol    = st.slider("Cholesterol (mg/dL)", 100, 400, 280)
    sugar   = st.slider("Blood Sugar (mg/dL)", 70, 300, 140)
    hr      = st.slider("Heart Rate (bpm)", 50, 200, 95)
    st.divider()
    st.subheader("Complaint & Imaging")
    complaint  = st.text_area(
        "Describe symptoms",
        "Severe chest pain and breathlessness since 2 hours with sweating"
    )
    xray_file  = st.file_uploader(
        "Upload Chest X-ray (optional)", type=["jpg", "jpeg", "png"]
    )
    run_btn    = st.button("🚀 Run AI Triage", type="primary", use_container_width=True)

# ── Main Panel ────────────────────────────────────────────────────
if not run_btn:
    st.info("👈 Fill in patient details in the sidebar and click **Run AI Triage**")
    st.image(
        "https://img.icons8.com/color/200/hospital.png",
        width=150
    )

if run_btn:
    img_bytes = xray_file.read() if xray_file else None

    with st.spinner("🔄 Running all AI modules — ML → DL → NLP → LLM..."):
        agent  = TriageAgent()
        result = agent.run(
            name, age, bp, chol, sugar, hr, sex_int,
            complaint, img_bytes
        )

    st.success("✅ Triage complete!")
    st.divider()

    # ── Metric Cards ─────────────────────────────────────────────
    st.subheader("📊 AI Triage Results")
    col1, col2, col3 = st.columns(3)

    risk_color = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}.get(
        result["risk"]["label"], "⚪"
    )
    urg_color  = {"CRITICAL": "🔴", "MODERATE": "🟡", "MILD": "🟢"}.get(
        result["nlp"]["urgency"], "⚪"
    )
    xray_color = "🔴" if result["xray"]["label"] == "PNEUMONIA" else "🟢"

    col1.metric(
        label="❤️ Cardiovascular Risk",
        value=f"{risk_color} {result['risk']['label']}",
        delta=f"{result['risk']['score']:.0%} probability"
    )
    col2.metric(
        label="🫁 X-Ray Finding",
        value=f"{xray_color} {result['xray']['label']}",
        delta=f"{result['xray']['confidence']:.0%} confidence"
    )
    col3.metric(
        label="⚡ Urgency Level",
        value=f"{urg_color} {result['nlp']['urgency']}",
    )

    st.divider()

    # ── Symptoms ──────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🩺 Extracted Symptoms")
        if result["nlp"]["symptoms"]:
            for s in result["nlp"]["symptoms"]:
                st.write(f"• {s.title()}")
        else:
            st.write("No specific symptoms extracted")

    with col_b:
        if xray_file:
            xray_file.seek(0)
            st.subheader("🔬 Uploaded X-Ray")
            st.image(xray_file, width=250)

    st.divider()

    # ── AI Clinical Report ────────────────────────────────────────
    st.subheader("📋 AI Clinical Summary")
    st.info(result["report"])

    # ── Patient Summary Card ──────────────────────────────────────
    st.divider()
    with st.expander("📄 Full Patient Summary"):
        st.write(f"**Name:** {result['name']}")
        st.write(f"**Age:** {result['age']} years")
        st.write(f"**BP:** {bp} mmHg  |  **Cholesterol:** {chol} mg/dL")
        st.write(f"**Blood Sugar:** {sugar} mg/dL  |  **Heart Rate:** {hr} bpm")
        st.write(f"**Complaint:** {complaint}")
        st.write(f"**Risk Score:** {result['risk']['score']:.2f}")
        st.write(f"**X-Ray Confidence:** {result['xray']['confidence']:.2f}")
