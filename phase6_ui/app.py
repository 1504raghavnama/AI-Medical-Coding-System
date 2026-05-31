
import streamlit as st
import spacy
import chromadb
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

st.set_page_config(
    page_title="AI Medical Coding System",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 AI-Powered Medical Coding System")
st.markdown("Automatically suggests ICD-10-CM codes from clinical notes using NLP + RAG + LLM")

@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_sci_lg")
    
    client = chromadb.PersistentClient(path="/kaggle/working/chromadb_store")
    collection = client.get_collection("icd10_2026")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        quantization_config=bnb_config,
        device_map="auto"
    )
    return nlp, collection, tokenizer, model

with st.spinner("Loading AI models... please wait"):
    nlp, collection, tokenizer, model = load_models()
st.success("Models loaded!")

non_medical = [
    "patient", "evidence", "history", "presentation",
    "complaint", "report", "finding", "result", "note"
]
abbreviations = {
    "pt": "patient", "hx": "history", "dx": "diagnosis",
    "rx": "prescription", "sx": "symptoms", "tx": "treatment",
    "sob": "shortness of breath", "bid": "twice daily",
    "tid": "three times daily", "prn": "as needed",
    "htn": "hypertension", "dm": "diabetes mellitus",
    "cad": "coronary artery disease", "chf": "congestive heart failure",
    "copd": "chronic obstructive pulmonary disease",
    "uti": "urinary tract infection", "mi": "myocardial infarction",
    "cva": "cerebrovascular accident", "t2dm": "type 2 diabetes mellitus",
    "ckd": "chronic kidney disease", "afib": "atrial fibrillation"
}

def preprocess(text):
    words = text.lower().split()
    expanded = []
    for w in words:
        clean_w = re.sub(r"[^\w]", "", w)
        expanded.append(abbreviations.get(clean_w, w))
    return nlp(" ".join(expanded))

def detect_temporal(doc):
    historical_words = ["history","previous","previously","former","formerly","past","old","prior"]
    uncertain_words = ["possible","possibly","probable","probably","suspected","rule out","query","questionable","likely"]
    negation_words = ["no","not","without","denies","denied","negative","absence","absent","never"]
    entities = []
    tokens = [token.text.lower() for token in doc]
    for ent in doc.ents:
        if ent.text.lower() in non_medical:
            continue
        start = max(0, ent.start - 7)
        window = " ".join(tokens[start:ent.start])
        if any(n in window for n in negation_words):
            status = "NEGATED"
        elif any(h in window for h in historical_words):
            status = "HISTORICAL"
        elif any(u in window for u in uncertain_words):
            status = "UNCERTAIN"
        else:
            status = "PRESENT"
        entities.append({"text": ent.text, "label": ent.label_, "status": status})
    return entities

def retrieve_codes(entity_text, n_results=5):
    results = collection.query(query_texts=[entity_text], n_results=n_results)
    codes = []
    for doc, meta, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        codes.append({"code": meta["code"], "description": doc, "confidence": round(1 - distance, 4)})
    return codes

def llm_suggest_code(clinical_note, entity, candidate_codes):
    candidates_text = "\n".join([
        f"  {i+1}. {c['code']} — {c['description']} (confidence: {c['confidence']})"
        for i, c in enumerate(candidate_codes)
    ])
    prompt = f"""You are an expert medical coder. Select the most appropriate ICD-10-CM code.
Clinical Note: {clinical_note}
Condition to code: {entity}
Candidate codes:
{candidates_text}
Select the single best code and explain why in one sentence.
Format: CODE: <code> | REASON: <reason>"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.1, do_sample=True)
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return response.strip()

# UI
st.subheader("Enter Clinical Note")
sample = "Pt with hx of mi and t2dm has htn and ckd. Possible afib. No sob."
clinical_note = st.text_area("Clinical Note", value=sample, height=150)

if st.button("🔍 Analyze & Suggest Codes", type="primary"):
    if clinical_note.strip():
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("NLP Analysis")
            doc = preprocess(clinical_note)
            entities = detect_temporal(doc)
            fixed = []
            for e in entities:
                if e["text"].startswith("no "):
                    e["text"] = e["text"].replace("no ", "")
                    e["status"] = "NEGATED"
                fixed.append(e)
            
            codeable = [e for e in fixed if e["status"] == "PRESENT"]
            non_codeable = [e for e in fixed if e["status"] != "PRESENT"]
            
            st.markdown("**✅ Codeable Conditions:**")
            for e in codeable:
                st.success(e["text"])
            
            st.markdown("**❌ Not Codeable:**")
            for e in non_codeable:
                st.warning(f"{e['text']} ({e['status']})")
        
        with col2:
            st.subheader("Suggested ICD-10 Codes")
            for entity in codeable:
                with st.spinner(f"Processing: {entity['text']}"):
                    candidates = retrieve_codes(entity["text"], n_results=5)
                    llm_output = llm_suggest_code(clinical_note, entity["text"], candidates)
                    code_match = re.search(r"CODE:\s*([A-Z0-9]+)", llm_output)
                    reason_match = re.search(r"REASON:\s*(.+)", llm_output)
                    
                    code = code_match.group(1) if code_match else "N/A"
                    reason = reason_match.group(1) if reason_match else llm_output
                    
                    st.markdown(f"**{entity['text'].title()}**")
                    st.info(f"📌 Code: `{code}`")
                    st.caption(f"Reason: {reason}")
                    st.divider()
    else:
        st.error("Please enter a clinical note.")

st.markdown("---")
st.caption("AI Medical Coding System | Internship Project | Built with spaCy + ChromaDB + Qwen")
