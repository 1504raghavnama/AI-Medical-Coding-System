
import spacy
import re

nlp = spacy.load("en_core_sci_lg")

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

def nlp_pipeline(clinical_note):
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
    return codeable, non_codeable
