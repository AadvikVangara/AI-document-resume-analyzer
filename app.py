"""
app.py
------
Modern Streamlit application for AI Document & Resume Intelligence.
Features:
- Premium Glassmorphic & Modern SaaS UI Design
- Document Parser (PDF, DOCX, TXT)
- Grounded Multi-turn Chatbot (powered by Gemini 3.6 Flash)
- Multi-style Smart Summarization
- Resume & ATS Compatibility Analyzer
- Interactive GenAI Knowledge Lab
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from document_parser import extract_text_from_file, compute_text_stats
import ai_service

# Page Configuration
st.set_page_config(
    page_title="DocuSense AI | Document & Resume Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling (CSS Injection)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Gradient Hero Title */
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        line-height: 1.2;
    }

    .hero-subtitle {
        color: #64748b;
        font-size: 1.1rem;
        font-weight: 400;
        line-height: 1.5;
        margin-bottom: 1.5rem;
    }

    /* Pill Badges */
    .pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        background: rgba(99, 102, 241, 0.12);
        color: #6366f1;
        border: 1px solid rgba(99, 102, 241, 0.25);
        margin-bottom: 0.75rem;
    }

    /* Glassmorphic Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }

    /* Feature Highlights Grid */
    .feature-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px;
        height: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 20px -8px rgba(99, 102, 241, 0.15);
        border-color: #c7d2fe;
    }
    .feature-icon {
        font-size: 1.8rem;
        margin-bottom: 10px;
    }
    .feature-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 6px;
    }
    .feature-desc {
        font-size: 0.88rem;
        color: #64748b;
        line-height: 1.45;
    }

    /* Stat Metric Boxes */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 16px 0;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0f172a;
    }

    /* Streamlit Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 8px;
        font-size: 0.95rem;
        font-weight: 600;
        padding: 0 18px;
        color: #475569;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(99, 102, 241, 0.1) !important;
        color: #4f46e5 !important;
        border-bottom: 3px solid #4f46e5 !important;
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #e2e8f0;
    }
    
    /* Code and Preformatted blocks */
    pre, code {
        font-family: 'Fira Code', monospace !important;
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
    st.markdown("### ⚡ **DocuSense AI**")
    st.caption("Document & Resume Intelligence Studio")
    st.markdown("---")

    # Section 1: Authentication
    st.markdown("#### 🔑 **Gemini API Key**")
    
    # Check Streamlit Cloud secrets first, then local environment variables
    cloud_secret = ""
    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            cloud_secret = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    env_api_key = cloud_secret or os.getenv("GEMINI_API_KEY", "")
    user_api_key = st.text_input(
        "API Key",
        value=env_api_key,
        type="password",
        placeholder="AIzaSy...",
        label_visibility="collapsed",
        help="Get a free Gemini API key from Google AI Studio: https://aistudio.google.com/app/apikey",
    )
    active_api_key = user_api_key.strip() or env_api_key

    if not active_api_key:
        st.warning("⚠️ API Key required for AI analysis.")
        st.markdown(
            "👉 [**Get Free Gemini API Key**](https://aistudio.google.com/app/apikey)",
            unsafe_allow_html=True,
        )
    else:
        st.success("🔒 API Key active & secured")

    st.markdown("---")

    # Section 2: Model Engine Selection
    st.markdown("#### 🤖 **AI Model Engine**")
    
    # Query available models if API key is active, or use safe defaults
    dynamic_options = ["gemini-3.5-flash-lite", "gemini-3.6-flash"]
    if active_api_key:
        if "cached_models" not in st.session_state or st.session_state.get("cached_key") != active_api_key:
            fetched = ai_service.get_available_models(active_api_key)
            if fetched:
                st.session_state.cached_models = fetched
                st.session_state.cached_key = active_api_key
        dynamic_options = st.session_state.get("cached_models", ["gemini-3.5-flash-lite", "gemini-3.6-flash"])

    model_options = list(dict.fromkeys(dynamic_options + ["Custom Model"]))

    model_choice = st.selectbox(
        "Gemini Model",
        options=model_options,
        index=0,
        label_visibility="collapsed",
        help="gemini-3.5-flash-lite is Google's ultra-fast, high-capacity model.",
    )
    if model_choice == "Custom Model":
        model_choice = st.text_input("Enter Model ID", value="gemini-3.5-flash-lite")

    st.markdown("---")

    # Section 3: Document Ingestion
    st.markdown("#### 📂 **Upload File**")
    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed",
        help="Upload any resume, research paper, contract, or notes to analyze.",
    )

    # Action Buttons
    col_demo, col_clear = st.columns(2)
    with col_demo:
        if st.button("📄 Load Demo", use_container_width=True, help="Load pre-built Senior Engineer resume"):
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
        if st.button("🗑️ Reset", use_container_width=True, help="Clear active document and chat"):
            st.session_state.doc_text = ""
            st.session_state.doc_name = ""
            st.session_state.doc_stats = {}
            st.session_state.chat_history = []
            st.session_state.summary_cache = {}
            st.session_state.ats_result = ""
            st.rerun()

    # Ingestion Processor
    if uploaded_file is not None and uploaded_file.name != st.session_state.doc_name:
        try:
            with st.spinner(f"Extracting content from '{uploaded_file.name}'..."):
                bytes_data = uploaded_file.read()
                text, stats = extract_text_from_file(bytes_data, uploaded_file.name)
                st.session_state.doc_text = text
                st.session_state.doc_name = uploaded_file.name
                st.session_state.doc_stats = stats
                st.session_state.chat_history = []
                st.session_state.summary_cache = {}
                st.session_state.ats_result = ""
                st.success(f"Loaded: {uploaded_file.name}")
                st.rerun()
        except Exception as e:
            st.error(f"Failed to read file: {e}")

    # Sidebar Footer Telemetry
    if st.session_state.doc_text:
        st.markdown("---")
        st.caption(f"📁 **Active Document:** `{st.session_state.doc_name}`")


# --- MAIN VIEW HEADER ---
st.markdown(
    '<div class="pill-badge"><span>⚡</span> Next-Gen Document Intelligence</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="hero-title">DocuSense AI Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Interactive document reasoning, structured executive summaries, and automated ATS resume matching powered by Google Gemini.</div>',
    unsafe_allow_html=True,
)

# Dynamic Telemetry Bar (Shown when document is loaded)
if st.session_state.doc_text:
    stats = st.session_state.doc_stats
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-card">
                <div class="metric-label">📑 Document Pages</div>
                <div class="metric-value">{stats.get('pages', 1)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">📝 Word Count</div>
                <div class="metric-value">{stats.get('words', 0):,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">🔢 Estimated Tokens</div>
                <div class="metric-value">~{stats.get('estimated_tokens', 0):,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">⏱️ Est. Reading Time</div>
                <div class="metric-value">{stats.get('reading_time_mins', 1)} min</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # Feature Showcase Grid for New Users
    st.markdown(
        """
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 24px 0;">
            <div class="feature-card">
                <div class="feature-icon">💬</div>
                <div class="feature-title">Grounded Q&A Chatbot</div>
                <div class="feature-desc">Converse with multi-page PDFs, research papers, or contracts with strict anti-hallucination source grounding.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📝</div>
                <div class="feature-title">Smart Summarizer</div>
                <div class="feature-desc">Extract executive briefs, key facts, takeaways, or FAQ lists in one click with instant Markdown export.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">Resume & ATS Match</div>
                <div class="feature-desc">Benchmark candidate resumes against job descriptions, reveal missing keywords, and get tailored interview questions.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- MAIN NAVIGATION TABS ---
tab_chat, tab_summary, tab_ats, tab_learn = st.tabs([
    "💬 Interactive Chat",
    "📝 Smart Summaries",
    "🎯 Resume & ATS Match",
    "💡 Knowledge Lab",
])


# ==========================================
# TAB 1: CHAT WITH DOCUMENT
# ==========================================
with tab_chat:
    st.markdown("### 💬 Grounded Document Q&A")
    st.caption("Ask questions about your document. Responses are strictly grounded in document context.")

    if not st.session_state.doc_text:
        st.info("👈 **Upload a document (PDF, DOCX, TXT) or click '📄 Load Demo' in the sidebar to start chatting.**")
    else:
        # Prompt Suggestions
        st.markdown("**💡 Quick Prompt Suggestions:**")
        col_q1, col_q2, col_q3 = st.columns(3)
        sample_query = None
        if col_q1.button("📌 Summarize core topics", key="sq1", use_container_width=True):
            sample_query = "Summarize the core topics and purpose of this document."
        if col_q2.button("🛠️ List technical skills & tools", key="sq2", use_container_width=True):
            sample_query = "List all technical skills, programming languages, and tools mentioned in this document."
        if col_q3.button("📈 Extract quantifiable metrics", key="sq3", use_container_width=True):
            sample_query = "What quantifiable metrics, percentages, or performance gains are mentioned?"

        st.markdown("---")

        # Display Chat History
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat Input Bar
        user_input = st.chat_input("Ask a question about the document...") or sample_query

        if user_input:
            if not active_api_key:
                st.error("Please provide your Gemini API Key in the left sidebar to chat.")
            else:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing document context with Gemini..."):
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
        st.info("👈 **Upload a document or click '📄 Load Demo' in the sidebar to generate summaries.**")
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
                st.error("Please enter your Gemini API Key in the sidebar.")
            else:
                with st.spinner(f"Synthesizing {summary_style}..."):
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
        st.info("👈 **Upload a resume or click '📄 Load Demo' in the sidebar to analyze ATS compatibility.**")
    else:
        st.markdown("#### Step 1: Target Job Description (Optional)")
        job_description = st.text_area(
            "Paste Target Job Description (JD)",
            height=140,
            placeholder="Paste the full job posting text here (requirements, tech stack, responsibilities)... Leave empty for general resume audit.",
        )

        sample_jd_btn = st.button("📋 Load Sample Senior Python/AI Engineer JD")
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
# TAB 4: KNOWLEDGE LAB (GENAI CONCEPTS)
# ==========================================
with tab_learn:
    st.markdown("### 💡 GenAI Architecture & Concept Knowledge Lab")
    st.caption("Understand the core engineering decisions behind modern document intelligence applications.")

    with st.expander("1. 🧠 In-Context Learning vs. Traditional RAG", expanded=True):
        st.markdown(
            """
            - **What is In-Context Learning?**
              Modern LLMs like **Gemini 3.6 Flash** feature massive context windows (up to **1M+ tokens** / ~1,500 pages).
              Instead of needing an expensive chunking, vector database, and embedding retrieval pipeline (RAG) for standard documents (resumes, contracts, research papers), we can directly inject the **entire document** into the prompt context!
            - **Why is this superior for single documents?**
              1. **Zero Context Loss**: Chunking splits tables and paragraphs arbitrarily. Direct context preserves full structural hierarchy.
              2. **Global Reasoning**: The model can compare Page 1 with Page 40 effortlessly.
              3. **Architectural Simplicity**: No vector databases (Chroma, Pinecone) or embedding synchronization needed.
            """
        )

    with st.expander("2. 🛡️ Groundedness & Preventing Hallucinations"):
        st.markdown(
            """
            - **The Problem**: LLMs generate words probabilistically and may hallucinate unstated details.
            - **The Solution**: We implement **Grounded System Instructions**:
              ```python
              system_instruction = (
                  "Answer questions strictly grounded in the document context. "
                  "If the document does not contain the information, explicitly say: "
                  "'Based on the uploaded document, I cannot find information regarding this.'"
              )
              ```
              We pair this with a low **temperature (`0.2`)** to minimize variance and maximize factual accuracy.
            """
        )

    with st.expander("3. 🔢 Tokens vs. Words vs. Characters"):
        st.markdown(
            """
            - **What is a Token?**
              LLMs read text broken into byte-pair encoded chunks called *tokens*.
            - **Standard Rule of Thumb**:
              - `1 token` $\\approx$ `0.75 words` in English.
              - `1,000 tokens` $\\approx$ `750 words`.
              - A standard 1-page resume is $\\approx$ `400-600 words` (`600-800 tokens`).
              - Gemini 3.6 Flash can hold over `1,000,000 tokens` in a single prompt!
            """
        )

    with st.expander("4. 🔄 Streamlit Session State & Reactive Lifecycle"):
        st.markdown(
            """
            - **The Streamlit Lifecycle**:
              Every time a user clicks a button or types text, Streamlit executes the Python script from top to bottom.
            - **Why `st.session_state` is essential**:
              Without `st.session_state`, your extracted document text and chat history would be wiped on every click!
              Storing `st.session_state.chat_history` and `st.session_state.doc_text` ensures continuous memory without re-parsing files or calling Gemini redundantly.
            """
        )
