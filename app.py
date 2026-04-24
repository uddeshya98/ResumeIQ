import streamlit as st
import PyPDF2
import google.generativeai as genai
import io
import os
import time
from dotenv import load_dotenv

load_dotenv()

# ── Page config ──────────────────────────────────────────────

st.set_page_config(
    page_title="ResumeIQ: AI-Powered ATS Scoring Engine",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main { padding-top: 2rem; }

.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.25rem;
    line-height: 1.2;
}

.hero-sub {
    font-size: 1rem;
    color: #64748b;
    margin-bottom: 2rem;
}

.score-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    color: white;
    text-align: center;
    margin-bottom: 1.5rem;
}

.score-number {
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
}

.score-label {
    font-size: 0.9rem;
    opacity: 0.85;
    margin-top: 4px;
}

.section-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.section-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 0.5rem;
}

.tag {
    display: inline-block;
    background: #f1f5f9;
    color: #475569;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.8rem;
    font-weight: 500;
    margin: 3px 2px;
}

.tag-green { background: #dcfce7; color: #166534; }
.tag-red   { background: #fee2e2; color: #991b1b; }
.tag-amber { background: #fef9c3; color: #854d0e; }

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 1.5rem;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.2s;
}

.stButton > button:hover { opacity: 0.9; }

.stFileUploader {
    border: 2px dashed #cbd5e1 !important;
    border-radius: 12px !important;
}

div[data-testid="stExpander"] {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────

def extract_text(uploaded_file) -> str:
    """Extract plain text from PDF or TXT uploads."""
    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        ).strip()
    return uploaded_file.read().decode("utf-8").strip()


def build_prompt(resume_text: str, job_role: str, focus: list) -> str:
    role_line  = f"Target Role: {job_role}" if job_role else "Target Role: General / Not specified"
    focus_line = ", ".join(focus) if focus else "All areas"

    return f"""You are a senior HR consultant and resume expert with 15+ years of experience.

Analyze the resume below and return your feedback in EXACTLY this format (no deviations):

---SCORE---
[A single integer 1-100 representing overall resume quality]

---SUMMARY---
[2-3 sentence executive summary of the resume's overall strength]

---STRENGTHS---
[3-5 bullet points starting with ✅]

---IMPROVEMENTS---
[4-6 actionable bullet points starting with 🔧, each with a specific fix]

---KEYWORDS---
[Comma-separated list of 6-10 ATS keywords found in the resume]

---MISSING---
[Comma-separated list of 4-6 keywords/skills likely missing for the target role]

---QUICK_WINS---
[3 specific, high-impact changes that can be made in under 30 minutes, numbered 1. 2. 3.]

{role_line}
Focus Areas: {focus_line}

Resume:
{resume_text}"""


def parse_response(raw: str) -> dict:
    """Parse Gemini's structured response into a dict."""
    sections = {
        "score":        "70",
        "summary":      "",
        "strengths":    "",
        "improvements": "",
        "keywords":     "",
        "missing":      "",
        "quick_wins":   "",
    }
    mapping = {
        "---SCORE---":        "score",
        "---SUMMARY---":      "summary",
        "---STRENGTHS---":    "strengths",
        "---IMPROVEMENTS---": "improvements",
        "---KEYWORDS---":     "keywords",
        "---MISSING---":      "missing",
        "---QUICK_WINS---":   "quick_wins",
    }

    current_key = None
    buffer      = []

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped in mapping:
            if current_key:
                sections[current_key] = "\n".join(buffer).strip()
            current_key = mapping[stripped]
            buffer      = []
        elif current_key:
            buffer.append(line)

    if current_key:
        sections[current_key] = "\n".join(buffer).strip()

    return sections


def analyze_resume(resume_text: str, job_role: str, focus: list) -> tuple:
    """Call Gemini API with retry logic and return parsed feedback dict."""
    
    # ✅ FIX 1: gemini-1.5-flash use karo (15 RPM free tier — much better)
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model  = genai.GenerativeModel("gemini-1.5-flash")
    prompt = build_prompt(resume_text, job_role, focus)

    # ✅ FIX 2: Retry logic — rate limit aaye toh auto 10 sec baad try karo
    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            return parse_response(response.text), response.text

        except Exception as e:
            error_str = str(e)
            if ("429" in error_str or "quota" in error_str.lower()) and attempt < 2:
                wait_time = (attempt + 1) * 10  # 10s → 20s
                st.warning(f"⏳ Rate limit hit. Auto-retrying in {wait_time}s... (attempt {attempt+1}/3)")
                time.sleep(wait_time)
            else:
                raise e


def score_color(score: int) -> str:
    if score >= 80:
        return "#16a34a"
    if score >= 60:
        return "#d97706"
    return "#dc2626"


# ── UI ────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">📄 ResumeIQ: AI-Powered ATS Scoring Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Get expert, AI-powered feedback on your resume in seconds.</div>', unsafe_allow_html=True)

# ✅ FIX 3: Session state — page reload pe dobara API call nahi hogi
if "result"      not in st.session_state:
    st.session_state.result       = None
if "raw_response" not in st.session_state:
    st.session_state.raw_response = None
if "analyzing"   not in st.session_state:
    st.session_state.analyzing    = False

# Input columns
col1, col2 = st.columns([3, 2])

with col1:
    uploaded_file = st.file_uploader(
        "Upload resume",
        type=["pdf", "txt"],
        help="PDF or plain text file",
        label_visibility="collapsed",
    )
    if uploaded_file:
        st.caption(f"✅ Uploaded: **{uploaded_file.name}**")

with col2:
    job_role = st.text_input(
        "Target job role (optional)",
        placeholder="e.g. Data Scientist, SWE",
    )
    focus_areas = st.multiselect(
        "Focus areas",
        ["Content & Clarity", "Skills Section", "Work Experience",
         "Formatting", "ATS Keywords", "Action Verbs"],
        default=["Content & Clarity", "ATS Keywords"],
    )

# ✅ FIX 4: Button spam rukta hai — analyzing ke waqt button disabled
analyze_clicked = st.button(
    "✨ Analyze Resume",
    disabled=(not uploaded_file or st.session_state.analyzing)
)

# ── Analysis ─────────────────────────────────────────────────

if analyze_clicked and uploaded_file:
    # Naya analysis — purana result clear karo
    st.session_state.result       = None
    st.session_state.raw_response = None
    st.session_state.analyzing    = True

    with st.spinner("Analyzing your resume... (this may take 10–20 seconds)"):
        try:
            resume_text = extract_text(uploaded_file)

            if not resume_text:
                st.error("Could not extract text from the file. Please try a different file.")
                st.session_state.analyzing = False
                st.stop()

            feedback, raw_response = analyze_resume(resume_text, job_role, focus_areas)

            # Session state mein save karo
            st.session_state.result       = feedback
            st.session_state.raw_response = raw_response

        except Exception as e:
            error_msg = str(e).lower()
            if "api_key" in error_msg or "authentication" in error_msg or "403" in error_msg or "400" in error_msg:
                st.error("❌ Invalid API key. Please check your GEMINI_API_KEY.")
            elif "429" in error_msg or "quota" in error_msg:
                st.error("⏳ Rate limit hit after 3 retries. Please wait 1–2 minutes and try again.")
            else:
                st.error(f"❌ Something went wrong: {str(e)}")

        finally:
            st.session_state.analyzing = False

# ── Display Results (session state se — reload safe) ──────────

if st.session_state.result:
    feedback     = st.session_state.result
    raw_response = st.session_state.raw_response

    score = int(feedback["score"].strip()) if feedback["score"].strip().isdigit() else 70
    color = score_color(score)

    # Score card
    st.markdown(f"""
    <div class="score-card">
        <div class="score-number">{score}<span style="font-size:1.5rem">/100</span></div>
        <div class="score-label">Overall Resume Score</div>
    </div>
    """, unsafe_allow_html=True)

    # Summary
    st.markdown(f"""
    <div class="section-card">
        <div class="section-title">Summary</div>
        <p style="margin:0; color:#334155; line-height:1.6">{feedback['summary']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Strengths + Improvements
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">✅ Strengths</div>', unsafe_allow_html=True)
        for line in feedback["strengths"].splitlines():
            if line.strip():
                st.markdown(line.strip())
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔧 Improvements</div>', unsafe_allow_html=True)
        for line in feedback["improvements"].splitlines():
            if line.strip():
                st.markdown(line.strip())
        st.markdown('</div>', unsafe_allow_html=True)

    # Keywords
    keywords = [k.strip() for k in feedback["keywords"].split(",") if k.strip()]
    missing  = [k.strip() for k in feedback["missing"].split(",")  if k.strip()]

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏷️ ATS Keywords Found</div>', unsafe_allow_html=True)
    tags_html = " ".join(f'<span class="tag tag-green">{k}</span>' for k in keywords)
    st.markdown(tags_html, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-title">⚠️ Likely Missing Keywords</div>', unsafe_allow_html=True)
    missing_html = " ".join(f'<span class="tag tag-red">{k}</span>' for k in missing)
    st.markdown(missing_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Quick wins
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚡ Quick Wins (30-min fixes)</div>', unsafe_allow_html=True)
    for line in feedback["quick_wins"].splitlines():
        if line.strip():
            st.info(line.strip())
    st.markdown('</div>', unsafe_allow_html=True)

    # Download
    st.download_button(
        label="⬇️ Download Full Analysis",
        data=raw_response,
        file_name="resume_analysis.txt",
        mime="text/plain",
    )

elif analyze_clicked and not uploaded_file:
    st.warning("Please upload a resume first.")
