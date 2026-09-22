"""
ai_service.py
-------------
GenAI service module handling all interactions with Google Gemini.
Supports both modern `google-genai` and legacy `google-generativeai` SDKs.
Contains specialized prompt templates for Document Chat, Deep Summarization,
and Resume / ATS Analysis.
"""

import os
from typing import List, Dict, Any, Generator, Union

# Attempt importing modern google-genai first; fallback to google-generativeai
HAVE_MODERN_GENAI = False
HAVE_LEGACY_GENAI = False

try:
    from google import genai
    from google.genai import types
    HAVE_MODERN_GENAI = True
except ImportError:
    try:
        import google.generativeai as legacy_genai
        HAVE_LEGACY_GENAI = True
    except ImportError:
        pass


def is_sdk_available() -> bool:
    """Checks if any Gemini SDK is installed and available."""
    return HAVE_MODERN_GENAI or HAVE_LEGACY_GENAI


def call_gemini(prompt: str, system_instruction: str = "", model_name: str = "gemini-2.5-flash", api_key: str = None) -> str:
    """
    Unified caller for Gemini API that abstracts SDK differences.
    """
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Google Gemini API Key is missing. Please provide it in the sidebar or via .env file.")

    # 1. Try modern google-genai SDK
    if HAVE_MODERN_GENAI:
        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            system_instruction=system_instruction if system_instruction else None,
            temperature=0.2,  # Low temperature for high factual accuracy
        )
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
        return response.text

    # 2. Fallback to legacy google-generativeai SDK
    elif HAVE_LEGACY_GENAI:
        legacy_genai.configure(api_key=api_key)
        model = legacy_genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction if system_instruction else None,
            generation_config={"temperature": 0.2},
        )
        response = model.generate_content(prompt)
        return response.text

    else:
        raise RuntimeError(
            "Neither `google-genai` nor `google-generativeai` is installed. Please run `pip install google-genai`."
        )


def chat_with_document(
    doc_text: str,
    chat_history: List[Dict[str, str]],
    user_query: str,
    model_name: str = "gemini-2.5-flash",
    api_key: str = None,
) -> str:
    """
    Answers user questions strictly based on the uploaded document text.
    Maintains groundedness and cites page numbers/sections if available.
    """
    system_instruction = (
        "You are an expert Document Analysis Assistant. "
        "Your task is to answer user questions accurately, strictly grounded in the provided document text.\n"
        "Guidelines:\n"
        "1. Base your answer ONLY on the provided document. Do NOT extrapolate or assume unstated facts.\n"
        "2. If the document does not contain enough information to answer the question, clearly state: "
        "'Based on the uploaded document, I cannot find information regarding this question.'\n"
        "3. Cite page numbers or sections when quoting or referencing specific data.\n"
        "4. Keep your responses clear, professional, and well-structured using markdown."
    )

    # Format previous conversation context (up to last 6 turns for conversational coherence)
    recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
    history_str = ""
    for msg in recent_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    prompt = f"""<DOCUMENT_CONTEXT>
{doc_text}
</DOCUMENT_CONTEXT>

<CONVERSATION_HISTORY>
{history_str}
</CONVERSATION_HISTORY>

<USER_QUESTION>
{user_query}
</USER_QUESTION>

Please answer the question based solely on the document context above:"""

    return call_gemini(prompt, system_instruction=system_instruction, model_name=model_name, api_key=api_key)


def generate_document_summary(
    doc_text: str,
    summary_type: str = "Executive Summary",
    model_name: str = "gemini-2.5-flash",
    api_key: str = None,
) -> str:
    """
    Generates a structured summary tailored to the requested style.
    """
    style_prompts = {
        "Executive Summary": (
            "Provide a high-level executive overview (2-3 paragraphs) capturing the main purpose, "
            "core findings or objectives, and major outcomes."
        ),
        "Key Bullet Points": (
            "Provide a concise, bulleted breakdown of the most critical facts, numbers, dates, "
            "and decisions outlined in the document."
        ),
        "Deep Analytical Summary": (
            "Provide a comprehensive, section-by-section breakdown:\n"
            "1. 🎯 Purpose & Scope\n"
            "2. 🔍 Core Findings / Key Data Points\n"
            "3. ⚠️ Risks, Challenges, or Gaps\n"
            "4. 🚀 Actionable Recommendations / Next Steps"
        ),
        "FAQ / Quick Reference": (
            "Formulate 5 to 7 key Questions & Answers that someone reading this document would need to know."
        ),
    }

    selected_instruction = style_prompts.get(summary_type, style_prompts["Executive Summary"])

    system_instruction = (
        "You are an expert Document Summarizer and Information Architect. "
        "Extract the most relevant, high-impact details without fluff or filler words."
    )

    prompt = f"""<DOCUMENT_CONTENT>
{doc_text}
</DOCUMENT_CONTENT>

<TASK>
{selected_instruction}
</TASK>

Format your output in clean, readable Markdown with headings, bold highlights, and bullet points where appropriate."""

    return call_gemini(prompt, system_instruction=system_instruction, model_name=model_name, api_key=api_key)


def analyze_resume_ats(
    resume_text: str,
    job_description: str = "",
    model_name: str = "gemini-2.5-flash",
    api_key: str = None,
) -> str:
    """
    Performs comprehensive Resume & ATS (Applicant Tracking System) evaluation.
    If Job Description is provided, evaluates role alignment and keyword match.
    If no JD is provided, performs a standalone resume audit.
    """
    system_instruction = (
        "You are an elite Tech Recruiter, Career Coach, and Applicant Tracking System (ATS) Expert. "
        "Your goal is to provide honest, constructive, actionable, and data-backed resume reviews."
    )

    if job_description.strip():
        prompt = f"""<RESUME>
{resume_text}
</RESUME>

<TARGET_JOB_DESCRIPTION>
{job_description}
</TARGET_JOB_DESCRIPTION>

<TASK>
Analyze the candidate's resume strictly against the target Job Description. 
Provide a structured evaluation in the following format:

## 📊 ATS Match Score
- **Estimated Match:** [0-100]%
- **Verdict:** [Strong Match / Moderate Match / Low Match]
- **Brief Rationale:** (2-3 sentences explaining the percentage)

---

## 🌟 Top Candidate Strengths
- Highlight 3-4 specific experiences, technologies, or achievements in the resume that strongly align with the JD.

---

## ⚠️ Critical Skill & Experience Gaps
- Identify key requirements or technologies from the JD that are completely missing or weak in the resume.

---

## 🔑 ATS Keyword Optimization
- **Keywords Found:** List matching keywords present in both.
- **Missing Keywords to Incorporate:** Crucial industry terms/skills from the JD that the candidate should add.

---

## 🛠️ Actionable Resume Fixes (Before & After)
- Provide 2 specific bullet points from the resume rewritten using the Google XYZ Formula: *"Accomplished [X] as measured by [Y], by doing [Z]"*.

---

## 🎯 Top 5 Tailored Interview Questions
- Generate 5 technical or behavioral interview questions specifically designed to test the candidate on the intersections between this resume and the target role.
"""
    else:
        prompt = f"""<RESUME>
{resume_text}
</RESUME>

<TASK>
Perform a standalone comprehensive ATS Audit and Professional Review of this resume.
Provide a structured evaluation in the following format:

## 📊 General ATS & Quality Rating
- **Overall Score:** [0-100]%
- **Resume Category / Seniority Estimate:** (e.g. Mid-Level Software Engineer, Entry-Level Data Analyst)
- **Summary Verdict:** (2-3 sentences on overall impact)

---

## 🌟 Identified Core Skills & Competencies
- **Technical Skills:** (Categorized by Languages, Frameworks, Tools, Cloud/Infra)
- **Domain Expertise & Methodologies:**

---

## 🚩 Areas for Improvement
- Highlight issues related to quantifiable metrics, impact statements, formatting, or passive phrasing.

---

## 🛠️ Bullet Point Optimization
- Select 2 weak bullet points from the resume and rewrite them into high-impact, quantifiable accomplishment statements.

---

## 🚀 Recommended Next Steps
- 3 clear, actionable steps the candidate should take to boost interview callback rates.
"""

    return call_gemini(prompt, system_instruction=system_instruction, model_name=model_name, api_key=api_key)
