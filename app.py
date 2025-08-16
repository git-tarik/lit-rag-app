import streamlit as st
import requests
from typing import List, Optional
import mimetypes
from uuid import uuid4
from pathlib import Path
from supabase import create_client, Client
import smtplib
from email.message import EmailMessage
from datetime import datetime
import csv
import os
import streamlit.components.v1 as components  # for textarea auto-focus

# ===== Page setup =====
st.set_page_config(page_title="LIT-RAG", page_icon="⚡", layout="wide")

# ===== Small helpers =====
def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return st.secrets[name]
    except Exception:
        return default

def post_to_backend(base_url: str, token: str, doc_url: str, questions: List[str]) -> requests.Response:
    endpoint = base_url.rstrip("/") + "/api/v1/hackrx/run"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"documents": doc_url, "questions": questions}
    return requests.post(endpoint, headers=headers, json=payload, timeout=120)

# ---- Supabase client (server-side only) ----
SB_URL = get_secret("SUPABASE_URL")
SB_SERVICE_KEY = get_secret("SUPABASE_SERVICE_ROLE_KEY")
SB_BUCKET = get_secret("SUPABASE_BUCKET", "docs")
sb: Optional[Client] = create_client(SB_URL, SB_SERVICE_KEY) if SB_URL and SB_SERVICE_KEY else None

def upload_to_supabase(uploaded_file) -> str:
    """Upload a file to Supabase Storage and return its public URL."""
    if not sb:
        raise RuntimeError("Supabase not configured. Add SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY to secrets.toml")
    file_bytes = uploaded_file.getvalue()
    ext = Path(uploaded_file.name).suffix.lower() or ".bin"
    mime = mimetypes.guess_type(uploaded_file.name)[0] or "application/octet-stream"
    path_in_bucket = f"uploads/{uuid4().hex}{ext}"

    res = sb.storage.from_(SB_BUCKET).upload(
        path=path_in_bucket,
        file=file_bytes,
        file_options={"content-type": mime, "cache-control": "3600"}  # no 'upsert' to avoid header bug
    )
    if hasattr(res, "status_code") and getattr(res, "status_code") and res.status_code >= 400:
        raise RuntimeError(f"Supabase upload failed: {getattr(res, 'text', '')}")

    pub = sb.storage.from_(SB_BUCKET).get_public_url(path_in_bucket)
    public_url = pub.get("publicUrl") if isinstance(pub, dict) else str(pub)
    if not public_url:
        raise RuntimeError("Could not obtain public URL from Supabase.")
    return public_url

def clean_questions(text: str) -> List[str]:
    qs = [line.strip() for line in text.splitlines() if line.strip()]
    return qs if qs else ([text.strip()] if text.strip() else [])

# ---- Feedback (SMTP or CSV fallback) ----
def send_feedback(message: str, user_email: str = "") -> bool:
    """
    If SMTP secrets exist, send an email to FEEDBACK_TO.
    Else, append to feedback.csv and return True.
    """
    smtp_host = get_secret("FEEDBACK_SMTP_SERVER")
    smtp_port = get_secret("FEEDBACK_SMTP_PORT")
    smtp_user = get_secret("FEEDBACK_SMTP_USER")
    smtp_pass = get_secret("FEEDBACK_SMTP_PASSWORD")
    feedback_to = get_secret("FEEDBACK_TO")

    if all([smtp_host, smtp_port, smtp_user, smtp_pass, feedback_to]):
        try:
            msg = EmailMessage()
            msg["Subject"] = "LIT-RAG Feedback"
            msg["From"] = smtp_user
            msg["To"] = feedback_to
            body = f"Feedback received at {datetime.utcnow().isoformat()}Z\n\nFrom: {user_email or '(not provided)'}\n\nMessage:\n{message}"
            msg.set_content(body)

            with smtplib.SMTP(smtp_host, int(smtp_port)) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
            return True
        except Exception as e:
            st.warning(f"Email send failed, saving locally instead. ({e})")

    # Fallback: save to CSV
    try:
        file_exists = os.path.isfile("feedback.csv")
        with open("feedback.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp_utc", "user_email", "message"])
            writer.writerow([datetime.utcnow().isoformat() + "Z", user_email, message])
        return True
    except Exception as e:
        st.error(f"Could not save feedback: {e}")
        return False

# ===== Prod/Dev config =====
APP_MODE = get_secret("APP_MODE", "prod")  # "prod" or "dev"
if APP_MODE == "dev":
    st.sidebar.title("Backend Settings (dev)")
    default_base = get_secret("BACKEND_BASE_URL", "")
    default_token = get_secret("API_BEARER_TOKEN", "")
    backend_base_url = st.sidebar.text_input("Backend Base URL", value=default_base, help="Your Railway FastAPI base URL")
    api_token = st.sidebar.text_input("Bearer Token", value=default_token, type="password")
    st.sidebar.caption("Loaded from .streamlit/secrets.toml (not committed).")
else:
    backend_base_url = get_secret("BACKEND_BASE_URL")
    api_token = get_secret("API_BEARER_TOKEN")
    # Hide the (empty) sidebar entirely
    st.markdown("<style>[data-testid='stSidebar']{display:none;}</style>", unsafe_allow_html=True)

if not backend_base_url or not api_token:
    st.error("Server misconfigured: missing BACKEND_BASE_URL or API_BEARER_TOKEN in secrets.")
    st.stop()

# ===== Styling (product header + responsive) =====
st.markdown(
    """
    <style>
      .max-container {max-width: 1200px; margin: 0 auto;}
      .hero {
        background: linear-gradient(135deg, #6E56CF 0%, #9b5de5 50%, #00c2ff 100%);
        color: #fff; padding: 28px 22px; border-radius: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15); margin-bottom: 18px;
      }
      .hero h1 { margin: 0; font-weight: 800; font-size: 42px; letter-spacing: .2px; }
      .hero p  { margin: 6px 0 0; font-size: 16px; opacity: .95; }
      .card {
        background: #ffffff; border: 1px solid rgba(0,0,0,.06);
        border-radius: 14px; padding: 18px; box-shadow: 0 8px 24px rgba(0,0,0,0.06);
      }
      .muted { color: rgba(0,0,0,.65); }
      /* Make inputs stretch nicely */
      .stTextInput>div>div>input, .stTextArea textarea { font-size: 15px; }
      @media (max-width: 640px){
        .hero h1 { font-size: 28px; }
        .hero p { font-size: 14px; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="max-container">
      <div class="hero">
        <h1>LIT-RAG</h1>
        <p>Ask your docs</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container():
    st.markdown('<div class="max-container">', unsafe_allow_html=True)

    # ===== Mode toggle: Upload (default) or URL =====
    mode = st.radio("Input mode", ["Upload file", "Use URL"], horizontal=True, index=0, label_visibility="collapsed")

    # ===== Upload mode (default) =====
    if mode == "Upload file":
        up = st.file_uploader(
            "Upload a file (PDF, DOCX, TXT, etc.)",
            type=["pdf", "docx", "txt", "md", "csv", "xlsx", "html"]
        )
        questions_text = st.text_area(
            "Questions",
            height=140,
            placeholder="Example:\nSummarize the report.\nList the key findings.\nWhat are the important dates?\nWho are the stakeholders?",
            help="Enter questions."
        )
        ask_upload = st.button("Ask (upload)", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

        if ask_upload:
            if not up:
                st.error("Please upload a file.")
            else:
                qs = clean_questions(questions_text)
                if not qs:
                    st.error("Please write at least one question.")
                else:
                    with st.spinner("Uploading file to Supabase Storage…"):
                        try:
                            temp_url = upload_to_supabase(up)
                        except Exception as e:
                            st.error("Upload failed.")
                            st.exception(e)
                            temp_url = None

                    if temp_url:
                        st.success(f"Uploaded ✓")
                        with st.spinner("Querying backend…"):
                            try:
                                r = post_to_backend(backend_base_url, api_token, temp_url, qs)
                                if r.status_code == 200:
                                    data = r.json()
                                    answers = data.get("answers", [])
                                    if answers:
                                        st.subheader("Answers")
                                        for i, ans in enumerate(answers, 1):
                                            st.markdown(f"{i}.** {ans}")
                                    else:
                                        st.info("No answers field in response. Full response:")
                                        st.json(data)
                                elif r.status_code in (401, 403):
                                    st.error("Auth failed (401/403). Check your Bearer token.")
                                elif r.status_code == 422:
                                    st.error("Validation error (422). Check payload format.")
                                    st.code(r.text)
                                else:
                                    st.error(f"Backend error: {r.status_code}")
                                    st.code(r.text)
                            except Exception as e:
                                st.exception(e)

    # ===== URL mode =====
    else:
        doc_url = st.text_input("Document URL", placeholder="https://…/your.pdf or .docx or .txt")
        questions_text2 = st.text_area(
            "Questions (one per line)",
            height=140,
            placeholder="Example:\nWhat is the policy change?\nList key dates mentioned.\nExtract the main conclusions.",
            help="Enter each question on a new line."
        )
        ask_url = st.button("Ask (URL)", type="primary")  # <-- ONLY CHANGE: make purple like upload
        st.markdown('</div>', unsafe_allow_html=True)

        if ask_url:
            if not doc_url:
                st.error("Please paste a document URL.")
            else:
                qs = clean_questions(questions_text2)
                if not qs:
                    st.error("Please write at least one question.")
                else:
                    with st.spinner("Querying backend…"):
                        try:
                            r = post_to_backend(backend_base_url, api_token, doc_url, qs)
                            if r.status_code == 200:
                                data = r.json()
                                answers = data.get("answers", [])
                                if answers:
                                    st.subheader("Answers")
                                    for i, ans in enumerate(answers, 1):
                                        st.markdown(f"{i}.** {ans}")
                                else:
                                    st.info("No answers field in response. Full response:")
                                    st.json(data)
                            elif r.status_code in (401, 403):
                                st.error("Auth failed (401/403). Check your Bearer token.")
                            elif r.status_code == 422:
                                st.error("Validation error (422). Check payload format.")
                                st.code(r.text)
                            else:
                                st.error(f"Backend error: {r.status_code}")
                                st.code(r.text)
                        except Exception as e:
                            st.exception(e)

    # ===== Feedback (email required + textarea auto-focus) =====


    # ===== Footer (below feedback) =====
    st.markdown(
        """
        <hr style="height:1px;border:none;background:linear-gradient(90deg,transparent,rgba(0,0,0,.18),transparent);margin: 16px 0 10px 0;">
        <div style="color:rgba(0,0,0,.65);padding-bottom:28px;">
          <strong>LIT-RAG</strong> • © 2025 • Lightning-fast document Q&A
          <div style="margin-top:6px;font-size:13.5px;">
            Ask questions about PDFs, DOCX, TXT and more — get concise answers from your own content.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)

