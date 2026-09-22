"""
app.py
------
Streamlit application for AI Document & Resume Intelligence.
Features:
- Document Parser (PDF, DOCX, TXT)
- Interactive Grounded Chatbot
- Multi-style Smart Summarization
- Resume & ATS Compatibility Analyzer
- Educational Learning Lab (GenAI Architecture Concepts)
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()

from document_parser import extract_text_from_file, compute_text_stats
import ai_service

# Page Configuration
st.set_page_config(
    page_title="AI Document & Resume Intelligence",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .stat-box {
        background-color: #f1f5f9;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #dbeafe;
        color: #1d4ed8;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""
if "doc_name" not in st.session_state:
    st.session_state.doc_name = ""
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "summary_cache" not in st.session_state:
    st.session_state.summary_cache = {}
if "ats_result" not in st.session_state:
    st.session_state.ats_result = ""


# --- SIDEBAR: Configuration & Controls ---
with st.sidebar:
    st.markdown("### ⚙️ Settings & Ingestion")

    # API Key Configuration
    env_api_key = os.getenv("GEMINI_API_KEY", "")
    user_api_key = st.text_input(
        "Google Gemini API Key",
        value=env_api_key,
        type="password",
        placeholder="AIzaSy...",
        help="Get a free Gemini API key from Google AI Studio: https://aistudio.google.com/",
    )
    active_api_key = user_api_key.strip() or env_api_key

    if not active_api_key:
        st.warning("⚠️ Enter your Gemini API key to activate AI features.")
        st.markdown(
            "👉 [Get Free Gemini API Key](https://aistudio.google.com/app/apikey)",
            unsafe_allow_html=True,
        )
    else:
        st.success("✅ API Key configured!", icon="🔒")

    st.markdown("---")

    # Model Selection
    model_choice = st.selectbox(
        "Gemini Model",
        options=["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        index=0,
        help="gemini-2.5-flash is fast, accurate, and cost-efficient for document analysis.",
    )

    st.markdown("---")

    # Document Upload
    st.markdown("#### 📂 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF, DOCX, or TXT file",
        type=["pdf", "docx", "txt"],
        help="Upload any document or resume to inspect, summarize, and chat with.",
    )

    # Quick Demo Loader
    col_demo, col_clear = st.columns(2)
    with col_demo:
        if st.button("📄 Load Demo Resume", use_container_width=True):
            sample_path = os.path.join(os.path.dirname(__file__), "sample_resume.txt")
            if os.path.exists(sample_path):
                with open(sample_path, "r", encoding="utf-8") as f:
                    sample_text = f.read()
                st.session_state.doc_text = sample_text
                st.session_state.doc_name = "sample_resume.txt"
                st.session_state.doc_stats = compute_text_stats(sample_text, page_count=1)
                st.session_state.chat_history = []
                st.session_state.summary_cache = {}
                st.session_state.ats_result = ""
                st.rerun()

    with col_clear:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.doc_text = ""
            st.session_state.doc_name = ""
            st.session_state.doc_stats = {}
            st.session_state.chat_history = []
            st.session_state.summary_cache = {}
            st.session_state.ats_result = ""
            st.rerun()

    # Process uploaded file
    if uploaded_file is not None and uploaded_file.name != st.session_state.doc_name:
        try:
            with st.spinner(f"Reading '{uploaded_file.name}'..."):
                bytes_data = uploaded_file.read()
                text, stats = extract_text_from_file(bytes_data, uploaded_file.name)
                st.session_state.doc_text = text
                st.session_state.doc_name = uploaded_file.name
                st.session_state.doc_stats = stats
                st.session_state.chat_history = []
                st.session_state.summary_cache = {}
                st.session_state.ats_result = ""
                st.success(f"Loaded: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Failed to read file: {e}")

    # Document Statistics Card
    if st.session_state.doc_text:
        st.markdown("---")
        st.markdown("#### 📊 Document Telemetry")
        stats = st.session_state.doc_stats
        c1, c2 = st.columns(2)
        c1.metric("Pages / Sections", stats.get("pages", 1))
        c2.metric("Word Count", f"{stats.get('words', 0):,}")
        
        c3, c4 = st.columns(2)
        c3.metric("Est. Tokens", f"~{stats.get('estimated_tokens', 0):,}")
        c4.metric("Reading Time", f"~{stats.get('reading_time_mins', 1)} min")


# --- MAIN VIEW HEADER ---
st.markdown('<div class="badge">GenAI Project Portfolio</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">📑 AI Document & Resume Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Intelligent document Q&A, deep multi-style summarization, and automated ATS resume matching powered by Google Gemini.</div>',
    unsafe_allow_html=True,
)

# Warning if no document is loaded
if not st.session_state.doc_text:
    st.info(
        "👋 **Welcome! To get started:**\n\n"
        "1. Enter your Gemini API Key in the left sidebar (or set `GEMINI_API_KEY` in `.env`).\n"
        "2. Upload a **PDF, DOCX, or TXT** file, or click **'📄 Load Demo Resume'** to test immediately!\n"
        "3. Switch between the tabs below to Chat, Summarize, or run an ATS Evaluation."
    )

# Main Navigation Tabs
tab_chat, tab_summary, tab_ats, tab_learn = st.tabs([
    "💬 Chat with Document",
    "📝 Smart Summaries",
    "🎯 Resume & ATS Analyzer",
    "💡 How It Works (Knowledge Lab)",
])


# ==========================================
# TAB 1: CHAT WITH DOCUMENT
# ==========================================
with tab_chat:
    st.markdown("### 💬 Grounded Document Q&A")
    st.caption("Ask anything about the document. Answers are strictly grounded in document text with citations.")

    if not st.session_state.doc_text:
        st.warning("Please upload a document or click 'Load Demo Resume' in the sidebar first.")
    else:
        # Prompt Suggestions
        st.markdown("**Suggested Quick Questions:**")
        col_q1, col_q2, col_q3 = st.columns(3)
        sample_query = None
        if col_q1.button("📌 Summarize the core topics", key="sq1", use_container_width=True):
            sample_query = "Summarize the core topics and purpose of this document."
        if col_q2.button("🛠️ What skills / tools are listed?", key="sq2", use_container_width=True):
            sample_query = "List all technical skills, programming languages, and tools mentioned in this document."
        if col_q3.button("📈 What are the quantifiable results?", key="sq3", use_container_width=True):
            sample_query = "What quantifiable metrics, percentages, or performance gains are mentioned?"

        # Display Chat History
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat Input Bar
        user_input = st.chat_input("Ask a question about the document...") or sample_query

        if user_input:
            if not active_api_key:
                st.error("Please provide a Gemini API Key in the sidebar to chat.")
            else:
                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                # Generate Assistant Response
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing document context..."):
                        try:
                            response_text = ai_service.chat_with_document(
                                doc_text=st.session_state.doc_text,
                                chat_history=st.session_state.chat_history[:-1],
                                user_query=user_input,
                                model_name=model_choice,
                                api_key=active_api_key,
                            )
                            st.markdown(response_text)
                            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                        except Exception as e:
                            st.error(f"Error communicating with Gemini: {str(e)}")


# ==========================================
# TAB 2: SMART SUMMARIES
# ==========================================
with tab_summary:
    st.markdown("### 📝 Multi-Style Document Summarizer")
    st.caption("Generate structured, high-signal summaries tailored to your objective.")

    if not st.session_state.doc_text:
        st.warning("Please upload a document or click 'Load Demo Resume' in the sidebar first.")
    else:
        col_ctrl1, col_ctrl2 = st.columns([3, 1])
        with col_ctrl1:
            summary_style = st.selectbox(
                "Select Summary Type",
                options=[
                    "Executive Summary",
                    "Key Bullet Points",
                    "Deep Analytical Summary",
                    "FAQ / Quick Reference",
                ],
                index=0,
            )
        with col_ctrl2:
            st.write("")
            st.write("")
            generate_btn = st.button("⚡ Generate Summary", use_container_width=True, type="primary")

        if generate_btn:
            if not active_api_key:
                st.error("Please provide a Gemini API Key in the sidebar.")
            else:
                with st.spinner(f"Generating {summary_style}..."):
                    try:
                        summary_output = ai_service.generate_document_summary(
                            doc_text=st.session_state.doc_text,
                            summary_type=summary_style,
                            model_name=model_choice,
                            api_key=active_api_key,
                        )
                        st.session_state.summary_cache[summary_style] = summary_output
                    except Exception as e:
                        st.error(f"Summarization error: {e}")

        # Render cached summary if exists
        cached = st.session_state.summary_cache.get(summary_style)
        if cached:
            st.markdown("---")
            st.markdown(cached)
            st.download_button(
                label="📥 Download Summary as Markdown",
                data=cached,
                file_name=f"{st.session_state.doc_name or 'document'}_summary.md",
                mime="text/markdown",
            )


# ==========================================
# TAB 3: RESUME & ATS ANALYZER
# ==========================================
with tab_ats:
    st.markdown("### 🎯 Resume & ATS Match Analyzer")
    st.caption("Evaluate resumes against Applicant Tracking Systems (ATS) standards and target job descriptions.")

    if not st.session_state.doc_text:
        st.warning("Please upload a resume (PDF/DOCX/TXT) or load the demo resume in the sidebar to use this feature.")
    else:
        st.markdown("#### Step 1: Target Job Description (Optional but Recommended)")
        job_description = st.text_area(
            "Paste Target Job Description (JD)",
            height=160,
            placeholder="Paste the full job posting text here (requirements, responsibilities, tech stack)... Leave blank for a standalone resume audit.",
        )

        sample_jd_btn = st.button("📋 Load Sample Software Engineer Job Description")
        if sample_jd_btn:
            job_description = (
                "Role: Senior Python / AI Engineer\n"
                "Requirements:\n"
                "- 4+ years Python development experience with FastAPI or Flask\n"
                "- Hands-on experience integrating Generative AI APIs (OpenAI or Gemini)\n"
                "- Experience with AWS cloud infrastructure (ECS, Lambda, S3, Docker)\n"
                "- Strong understanding of relational databases (PostgreSQL) and caching (Redis)\n"
                "- Experience with vector databases and RAG architectures is a plus\n"
                "- Proven track record of improving system latency and scaling backend systems"
            )
            st.session_state["sample_jd_text"] = job_description
            st.rerun()

        # Check if sample JD is stored
        if "sample_jd_text" in st.session_state and not job_description:
            job_description = st.session_state["sample_jd_text"]

        run_ats = st.button("🚀 Analyze Resume & Match Against JD", type="primary")

        if run_ats:
            if not active_api_key:
                st.error("Please enter your Gemini API Key in the sidebar.")
            else:
                with st.spinner("Running deep ATS evaluation with Gemini..."):
                    try:
                        ats_result = ai_service.analyze_resume_ats(
                            resume_text=st.session_state.doc_text,
                            job_description=job_description,
                            model_name=model_choice,
                            api_key=active_api_key,
                        )
                        st.session_state.ats_result = ats_result
                    except Exception as e:
                        st.error(f"ATS Analysis error: {e}")

        if st.session_state.ats_result:
            st.markdown("---")
            st.markdown(st.session_state.ats_result)
            st.download_button(
                label="📥 Download ATS Report (.md)",
                data=st.session_state.ats_result,
                file_name=f"{st.session_state.doc_name or 'resume'}_ats_report.md",
                mime="text/markdown",
            )


# ==========================================
# TAB 4: HOW IT WORKS (KNOWLEDGE LAB)
# ==========================================
with tab_learn:
    st.markdown("### 💡 GenAI Architecture & Concept Knowledge Lab")
    st.caption("Learn the inner workings of LLMs, context windows, and modern AI engineering.")

    with st.expander("1. 🧠 In-Context Learning vs. Traditional RAG", expanded=True):
        st.markdown(
            """
            - **What is In-Context Learning?**
              Modern LLMs like **Gemini 1.5/2.5** feature massive context windows (up to **1 million to 2 million tokens**).
              Instead of needing a complicated chunking, vector database, and embedding retrieval pipeline (RAG) for standard documents (resumes, contracts, research papers, 200-page manuals), we can directly inject the **entire document** into the model's context!
            - **Why is this better for single documents?**
              1. **Zero Information Loss**: Chunking often cuts sentences or tables across boundaries. Direct context preserves full document hierarchy.
              2. **Global Reasoning**: The model can compare Page 1 with Page 50 seamlessly.
              3. **Simpler Architecture**: No vector database (Chroma, Pinecone) or embedding cost needed for documents under ~1M tokens.
            """
        )

    with st.expander("2. 🛡️ Groundedness & Preventing Hallucinations"):
        st.markdown(
            """
            - **The Hallucination Challenge**: LLMs are probabilistic autocomplete engines. When asked about facts they don't know, they may invent plausible-sounding details.
            - **How we solve it in this app**:
              We apply **Grounded System Prompting**:
              ```python
              system_instruction = (
                  "Answer questions strictly grounded in the document context. "
                  "If the document does not contain the information, explicitly say: "
                  "'Based on the uploaded document, I cannot find this information.'"
              )
              ```
              We also set a **low temperature (`0.2`)** to minimize randomness and maximize factual precision.
            """
        )

    with st.expander("3. 🔢 Tokens vs. Words vs. Characters"):
        st.markdown(
            """
            - **What is a Token?**
              LLMs do not read words or letters; they process chunks of characters called *tokens*.
            - **Quick Math Rules of Thumb**:
              - `1 token` $\\approx$ `0.75 words` in English.
              - `1,000 tokens` $\\approx$ `750 words`.
              - A standard 1-page resume is $\\approx$ `400-600 words` (around `600-800 tokens`).
              - Gemini 2.5 Flash has a context window of **1,048,576 tokens**, meaning it can read over **1,500 pages** in a single prompt!
            """
        )

    with st.expander("4. 🔄 Streamlit Session State & Re-execution Lifecycle"):
        st.markdown(
            """
            - **How Streamlit Works**:
              Every time a user clicks a button, types in an input, or switches a dropdown, Streamlit re-runs the entire Python script from top to bottom.
            - **Why `st.session_state` is critical**:
              Without `st.session_state`, your uploaded document, parsed text, and chat history would be wiped clean on every user click!
              By storing `st.session_state.chat_history` and `st.session_state.doc_text`, our state persists seamlessly across user interactions.
            """
        )
