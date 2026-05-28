# AI-Powered Medical Coding System

Internship project building an NLP + LLM + RAG pipeline for automated ICD-10-CM and CPT medical code suggestion from clinical notes.

## Architecture
- **Layer 2:** NLP & Information Extraction (spaCy, SciSpacy, Bio_ClinicalBERT)
- **Layer 3:** Coding Engine (Qwen LLM + RAG with ChromaDB)
- **Layer 4:** Knowledge Base (ICD-10, CPT, SNOMED CT)
- **Layer 6:** Human Review UI (Streamlit)

## Project Phases
| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Collection & Vector DB Setup | 🔄 In Progress |
| 2 | NLP Pipeline | ⏳ Pending |
| 3 | Entity Normalization | ⏳ Pending |
| 4 | RAG + LLM Integration | ⏳ Pending |
| 5 | Rule Engine & Validation | ⏳ Pending |
| 6 | Streamlit UI & Demo | ⏳ Pending |

## Tech Stack
- Python, HuggingFace Transformers, spaCy, SciSpacy
- ChromaDB (Vector Database)
- Qwen2.5-3B-Instruct (4-bit quantized)
- Streamlit + ngrok
- Kaggle GPU T4 x2
