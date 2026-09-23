# 📑 AI Document & Resume Intelligence Platform

A full-featured, production-ready Generative AI application built with **Python**, **Google Gemini API**, and **Streamlit**. 

Upload any document (PDF, Word DOCX, TXT) to chat interactively with grounded citations, extract multi-style executive summaries, or run an automated **ATS Resume Match & Audit** against target job descriptions.

---

## 🌟 Key Features

1. **💬 Grounded Document Q&A (Chatbot)**:
   - Chat with multi-page documents in real-time.
   - Grounded system instructions prevent hallucinations and enforce source citations.
   - Conversation history memory managed via Streamlit session state.

2. **📝 Multi-Style Smart Summarizer**:
   - One-click summaries: *Executive Brief*, *Key Bullet Points*, *Deep Analytical Breakdown*, and *FAQ / Quick Reference*.
   - Instant export and download as Markdown (`.md`).

3. **🎯 Resume & ATS Compatibility Analyzer**:
   - Compares candidate resumes against real Job Descriptions (JDs).
   - Generates an estimated **ATS Match Score (0-100%)**, identified strengths, missing keywords, bullet point rewrites (using Google's XYZ formula), and tailored interview questions.

4. **💡 Built-in GenAI Knowledge Lab**:
   - Interactive explanations inside the app covering **In-Context Learning vs RAG**, **Tokenization Math**, **Temperature controls**, and **Streamlit lifecycle**.

---

## 🏗️ Project Architecture

```
ai-doc-analyzer/
├── .venv/                   # Isolated Python virtual environment
├── .env.example             # Template for API keys
├── .gitignore               # Excludes secrets, cache, and virtual environment
├── requirements.txt         # Project dependencies
├── sample_resume.txt        # Demo resume for immediate 1-click testing
├── document_parser.py       # Text extraction engine (PDF, DOCX, TXT) & statistics
├── ai_service.py            # Gemini API wrapper, system prompts & intelligence layer
├── app.py                   # Streamlit interactive UI & session state logic
└── README.md                # Project documentation & GitHub guide
```

---

## 🚀 Quick Start (Run Locally)

### 1. Prerequisites
- Python 3.10+ installed on your machine.
- A free Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 2. Activate Virtual Environment & Install Dependencies
Open your terminal (PowerShell, Command Prompt, or Bash) in this project folder:

**Windows (PowerShell):**
```powershell
# Create virtual environment (if not already created)
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install required packages
pip install -r requirements.txt
```

**Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Your Gemini API Key
You have two easy options:
- **Option A (In-App)**: Paste your key directly into the sidebar text box when the app starts.
- **Option B (Environment File)**: Create a `.env` file from the template:
  ```bash
  cp .env.example .env
  ```
  Edit `.env` and insert your key:
  ```env
  GEMINI_API_KEY=AIzaSyYourActualKeyHere
  ```

### 4. Launch the Application
```bash
streamlit run app.py
```
Your default browser will automatically open: `http://localhost:8501` 🎉

---

## 🧠 Core GenAI Concepts Explained

### 1. In-Context Learning vs. Traditional RAG
- Traditional RAG (Retrieval-Augmented Generation) requires chopping documents into small chunks, creating numerical vector embeddings, and querying a vector database.
- Modern Gemini models have context windows up to **1M+ tokens** (over 1,500 pages). For single-document analysis, passing the entire document directly into context avoids boundary loss, preserves table formats, and eliminates vector database infrastructure costs.

### 2. Hallucination Prevention via Grounding
- LLMs generate probabilities, not absolute truths.
- In `ai_service.py`, we enforce strict grounding:
  ```python
  system_instruction = (
      "Base your answer ONLY on the provided document. "
      "If the document does not contain enough information, state: "
      "'Based on the uploaded document, I cannot find information regarding this question.'"
  )
  ```
  We combine this with a low **temperature (0.2)** to make the model strictly factual.

### 3. ATS Scoring & Resume Optimization
- Applicant Tracking Systems scan resumes for specific semantic keywords, quantifiable impact, and clear role alignment.
- The analyzer evaluates candidate experience using Google's **XYZ Formula**:
  > *"Accomplished [X] as measured by [Y], by doing [Z]"*
- It cross-references job requirements to highlight exact keyword gaps and generates tailored interview prep questions.

---

## 🛡️ License
This project is open-source and available under the [MIT License](LICENSE).
