import streamlit as st
import time
import joblib
import numpy as np
from scipy.sparse import hstack
from datetime import datetime
import sqlite3
import hashlib
import secrets
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def render_html(html, **kwargs):
    """Render HTML without displaying the HTML source code."""
    if hasattr(st, "html"):
        st.html(html)
    else:
        st.markdown(html, unsafe_allow_html=True)




# ============================================================
# DATABASE
# ============================================================

DB_NAME = "hiresafe.db"


def get_db():
    return sqlite3.connect(DB_NAME)


def init_database():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            company TEXT NOT NULL,
            job_title TEXT NOT NULL,
            result TEXT NOT NULL,
            confidence REAL NOT NULL,
            analyzed_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


def hash_password(password):
    salt = secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000
    ).hex()

    return f"{salt}${password_hash}"


def verify_password(password, stored_password):
    try:
        salt, stored_hash = stored_password.split("$", 1)

        password_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            100000
        ).hex()

        return secrets.compare_digest(password_hash, stored_hash)

    except ValueError:
        return False


def create_user(name, email, password):
    conn = get_db()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO users
            (name, email, password_hash, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                name.strip(),
                email.strip().lower(),
                hash_password(password),
                datetime.now().isoformat(timespec="seconds")
            )
        )

        conn.commit()
        user_id = cursor.lastrowid
        return user_id, None

    except sqlite3.IntegrityError:
        return None, "An account with this email already exists."

    finally:
        conn.close()


def authenticate_user(email, password):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, name, email, password_hash
        FROM users
        WHERE email = ?
        """,
        (email.strip().lower(),)
    )

    user = cursor.fetchone()
    conn.close()

    if user and verify_password(password, user[3]):
        return {
            "id": user[0],
            "name": user[1],
            "email": user[2]
        }

    return None


def save_analysis(user_id, company, job_title, result, confidence):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO analyses
        (user_id, company, job_title, result, confidence, analyzed_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            company.strip() or "Company not provided",
            job_title.strip(),
            result,
            float(confidence),
            datetime.now().isoformat(timespec="seconds")
        )
    )

    conn.commit()
    conn.close()


def get_user_analyses(user_id):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT company, job_title, result, confidence, analyzed_at
        FROM analyses
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (user_id,)
    )

    rows = cursor.fetchall()
    conn.close()

    return rows


init_database()

# ============================================================
# RESULT HELPERS
# ============================================================

# Keep the same suspicious-keyword list used during model training.
SUSPICIOUS_KEYWORDS = [
    "urgent",
    "immediate",
    "quick",
    "easy",
    "earn",
    "income",
    "bonus",
    "limited",
    "guaranteed",
    "apply now",
    "work from home",
    "no experience",
    "investment",
    "payment",
    "registration fee",
    "click"
]

# IMPORTANT: this preprocessing matches main.py exactly.
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))

try:
    LEMMATIZER = WordNetLemmatizer()
    LEMMATIZER.lemmatize("test")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    LEMMATIZER = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\+?\d[\d\s-]{8,}\d", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    words = []
    for word in text.split():
        if word not in STOP_WORDS:
            words.append(LEMMATIZER.lemmatize(word))
    return " ".join(words)

def keyword_score(text):
    text = str(text).lower()
    return sum(1 for word in SUSPICIOUS_KEYWORDS if word in text)

def get_suspicious_keywords(title, description):
    text = f"{title} {description}".lower()
    return [word for word in SUSPICIOUS_KEYWORDS if word in text]


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HireSafe AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# SESSION STATE
# ============================================================

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "user_email" not in st.session_state:
    st.session_state.user_email = ""

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# Load the already-trained model and TF-IDF vectorizer.
best_model = joblib.load("best_linear_svm.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")


# ============================================================
# THEME
# ============================================================

dark = st.session_state.theme == "dark"

if dark:
    BG = "#050505"
    CARD = "#111111"
    TEXT = "#F5F5F5"
    MUTED = "#9CA3AF"
    BORDER = "#272727"
    ACCENT = "#38BDF8"
else:
    BG = "#FFFFFF"
    CARD = "#F8FAFC"
    TEXT = "#111827"
    MUTED = "#6B7280"
    BORDER = "#E5E7EB"
    ACCENT = "#2563EB"


# ============================================================
# GLOBAL CSS
# ============================================================

render_html(
    f"""
<style>

.stApp {{
    background: {BG};
    color: {TEXT};
}}

.block-container {{
    padding-top: 1rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}}

#MainMenu {{
    visibility: hidden;
}}

header {{
    visibility: hidden;
}}

footer {{
    visibility: hidden;
}}


/* ---------------- NAVIGATION ---------------- */

.nav-logo {{
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -1px;
    color: {TEXT};
    padding-top: 8px;
}}

.nav-logo span {{
    color: {ACCENT};
}}


/* ---------------- HERO ---------------- */

.hero-container {{
    text-align: center;
    padding: 75px 20px 55px 20px;
}}

.hero-shield {{
    font-size: 75px;
    margin-bottom: 15px;
    display: inline-block;
    animation: shieldFloat 3s ease-in-out infinite;
}}

.hero-title {{
    font-size: 68px;
    font-weight: 900;
    letter-spacing: -4px;
    margin: 0;
    color: {TEXT};
}}

.hero-title span {{
    color: {ACCENT};
}}

.hero-subtitle {{
    margin-top: 18px;
    font-size: 21px;
    color: {MUTED};
}}

.hero-description {{
    max-width: 650px;
    margin: 18px auto;
    font-size: 16px;
    line-height: 1.7;
    color: {MUTED};
}}


/* ---------------- FEATURE CARDS ---------------- */

.feature {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 25px;
    min-height: 145px;
    transition: all 0.3s ease;
}}

.feature:hover {{
    transform: translateY(-6px);
    border-color: {ACCENT};
}}

.feature-icon {{
    font-size: 30px;
}}

.feature-title {{
    margin-top: 10px;
    font-size: 18px;
    font-weight: 700;
    color: {TEXT};
}}

.feature-text {{
    margin-top: 7px;
    font-size: 14px;
    line-height: 1.5;
    color: {MUTED};
}}


/* ---------------- BUTTONS ---------------- */

.stButton > button {{
    border-radius: 12px;
    border: 1px solid {BORDER};
    background: {CARD};
    color: {TEXT};
    font-weight: 600;
    transition: all 0.3s ease;
}}

.stButton > button:hover {{
    border-color: {ACCENT};
    color: {ACCENT};
}}


/* ---------------- ANIMATION ---------------- */

@keyframes shieldFloat {{

    0% {{
        transform: rotateY(0deg) translateY(0px);
    }}

    25% {{
        transform: rotateY(90deg) translateY(-8px);
    }}

    50% {{
        transform: rotateY(180deg) translateY(0px);
    }}

    75% {{
        transform: rotateY(270deg) translateY(-8px);
    }}

    100% {{
        transform: rotateY(360deg) translateY(0px);
    }}

}}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# 5 SECOND INTRO

# ============================================================

if "intro_seen" not in st.session_state:
    st.session_state.intro_seen = False

if not st.session_state.intro_seen:
    intro_placeholder = st.empty()

    intro_html = f"""
    <style>
        .hs-intro {{
            position: fixed;
            inset: 0;
            z-index: 999999;
            background: {BG};
            color: {TEXT};
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
        }}

        .hs-shield {{
            font-size: 100px;
            line-height: 1;
            display: inline-block;
            transform-style: preserve-3d;
            animation: hsRotate 2s linear infinite,
                       hsGlow 1.8s ease-in-out infinite;
        }}

        .hs-title {{
            margin-top: 28px;
            font-size: clamp(42px, 7vw, 72px);
            font-weight: 900;
            letter-spacing: -4px;
            opacity: 0;
            animation: hsFadeUp 1s ease forwards;
            animation-delay: .5s;
        }}

        .hs-title span {{
            color: {ACCENT};
        }}

        .hs-subtitle {{
            margin-top: 12px;
            font-size: 18px;
            color: {MUTED};
            opacity: 0;
            animation: hsFadeUp 1s ease forwards;
            animation-delay: 1.2s;
        }}

        .hs-loading {{
            margin-top: 30px;
            font-size: 13px;
            color: {ACCENT};
            opacity: 0;
            animation: hsFadeUp 1s ease forwards;
            animation-delay: 1.8s;
        }}

        .hs-loader {{
            width: 190px;
            height: 3px;
            margin-top: 12px;
            background: {BORDER};
            border-radius: 20px;
            overflow: hidden;
        }}

        .hs-loader-bar {{
            width: 0;
            height: 100%;
            background: {ACCENT};
            animation: hsLoad 4.2s linear forwards;
        }}

        @keyframes hsRotate {{
            0%   {{ transform: rotateY(0deg); }}
            50%  {{ transform: rotateY(180deg) scale(1.05); }}
            100% {{ transform: rotateY(360deg); }}
        }}

        @keyframes hsGlow {{
            0%,100% {{ filter: drop-shadow(0 0 5px {ACCENT}); }}
            50% {{ filter: drop-shadow(0 0 30px {ACCENT}); }}
        }}

        @keyframes hsFadeUp {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        @keyframes hsLoad {{
            from {{ width: 0%; }}
            to {{ width: 100%; }}
        }}
    </style>

    <div class="hs-intro">
        <div class="hs-shield">🛡️</div>
        <div class="hs-title">HIRE<span>SAFE</span> AI</div>
        <div class="hs-subtitle">Protecting Careers with Artificial Intelligence</div>
        <div class="hs-loading">Initializing AI Engine...</div>
        <div class="hs-loader">
            <div class="hs-loader-bar"></div>
        </div>
    </div>
    """

    # st.html renders the markup as HTML instead of displaying the tags as text.
    if hasattr(st, "html"):
        intro_placeholder.html(intro_html)
    else:
        intro_placeholder.markdown(intro_html, unsafe_allow_html=True)

    time.sleep(5)
    st.session_state.intro_seen = True
    st.rerun()


# NAVIGATION
# ============================================================

st.markdown("---")

col1, col2, col3, col4, col5, col6, col7 = st.columns(
    [3, 1, 1, 1, 1, 0.7, 0.7]
)

with col1:
    render_html(
        f"""
        <div class="nav-logo">
            🛡️ Hire<span>Safe</span> AI
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    if st.button("Home"):
        st.session_state.page = "Home"
        st.rerun()

with col3:
    if st.button("Analyze"):
        st.session_state.page = "Analyze"
        st.rerun()

with col4:
    if st.button("Dashboard"):
        st.session_state.page = "Dashboard"
        st.rerun()

with col5:
    if st.button("About"):
        st.session_state.page = "About"
        st.rerun()

with col6:
    if st.button("☀️" if dark else "🌙"):
        st.session_state.theme = "light" if dark else "dark"
        st.rerun()

with col7:
    if hasattr(st, "popover"):
        with st.popover("👤"):
            if st.session_state.logged_in:

                st.markdown(f"### {st.session_state.user_name}")
                st.caption(st.session_state.user_email)

                if st.button(
                    "Sign out",
                    use_container_width=True,
                    key="nav_signout"
                ):
                    st.session_state.logged_in = False
                    st.session_state.user_name = ""
                    st.session_state.user_email = ""
                    st.session_state.user_id = None
                    st.session_state.page = "Home"
                    st.rerun()

            else:

                sign_in_tab, sign_up_tab = st.tabs(
                    ["Sign In", "Sign Up"]
                )

                with sign_in_tab:
                    st.markdown("### Welcome back")

                    login_email = st.text_input(
                        "Email",
                        key="nav_login_email"
                    )

                    login_password = st.text_input(
                        "Password",
                        type="password",
                        key="nav_login_password"
                    )

                    if st.button(
                        "Sign In",
                        use_container_width=True,
                        key="nav_signin"
                    ):

                        if not login_email.strip() or not login_password:
                            st.warning(
                                "Please enter your email and password."
                            )
                        else:
                            user = authenticate_user(
                                login_email,
                                login_password
                            )

                            if user:
                                st.session_state.logged_in = True
                                st.session_state.user_id = user["id"]
                                st.session_state.user_name = user["name"]
                                st.session_state.user_email = user["email"]
                                st.session_state.page = "Dashboard"
                                st.rerun()
                            else:
                                st.error(
                                    "Invalid email or password."
                                )

                with sign_up_tab:
                    st.markdown("### Create your account")

                    signup_name = st.text_input(
                        "Name",
                        key="nav_signup_name"
                    )

                    signup_email = st.text_input(
                        "Email",
                        key="nav_signup_email"
                    )

                    signup_password = st.text_input(
                        "Password",
                        type="password",
                        key="nav_signup_password"
                    )

                    signup_confirm = st.text_input(
                        "Confirm Password",
                        type="password",
                        key="nav_signup_confirm"
                    )

                    if st.button(
                        "Create Account",
                        use_container_width=True,
                        key="nav_signup"
                    ):

                        if not all([
                            signup_name.strip(),
                            signup_email.strip(),
                            signup_password,
                            signup_confirm
                        ]):
                            st.warning(
                                "Please fill in all fields."
                            )

                        elif signup_password != signup_confirm:
                            st.error(
                                "Passwords do not match."
                            )

                        elif len(signup_password) < 6:
                            st.warning(
                                "Password must be at least 6 characters."
                            )

                        else:
                            user_id, error = create_user(
                                signup_name,
                                signup_email,
                                signup_password
                            )

                            if error:
                                st.error(error)

                            else:
                                st.session_state.logged_in = True
                                st.session_state.user_id = user_id
                                st.session_state.user_name = signup_name.strip()
                                st.session_state.user_email = signup_email.strip().lower()
                                st.session_state.page = "Dashboard"
                                st.rerun()

    else:
        if st.button("👤", key="profile_button"):
            st.session_state.page = "Profile"
            st.rerun()



# ANALYZE JOB PAGE
# ============================================================

if st.session_state.page == "Analyze":

    render_html(
        f"""
        <div style="
            text-align:center;
            padding:55px 20px 25px 20px;
        ">
            <div style="
                font-size:48px;
                margin-bottom:12px;
            ">🛡️</div>

            <div style="
                font-size:42px;
                font-weight:900;
                letter-spacing:-2px;
                color:{TEXT};
            ">
                Check a Job Posting
            </div>

            <div style="
                margin-top:12px;
                color:{MUTED};
                font-size:17px;
            ">
                Review a job before you decide to apply.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:
        job_title = st.text_input(
            "Job Title",
            placeholder="e.g. Data Analyst"
        )

        company_name = st.text_input(
            "Company Name",
            placeholder="e.g. Example Technologies"
        )

        location = st.text_input(
            "Location",
            placeholder="e.g. Hyderabad, India"
        )

        job_description = st.text_area(
            "Job Description",
            placeholder="Paste the complete job description here...",
            height=220
        )

    with right:
        requirements = st.text_area(
            "Requirements",
            placeholder="Paste the required skills, qualifications and experience...",
            height=160
        )

        benefits = st.text_area(
            "Benefits",
            placeholder="Paste the benefits, salary details or other information...",
            height=160
        )

        telecommuting = st.checkbox("Remote / Work From Home")
        has_company_logo = st.checkbox("Company Logo Available")
        has_questions = st.checkbox("Screening Questions Available")

    st.markdown("<br>", unsafe_allow_html=True)

    analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])

    with analyze_col2:
        analyze_clicked = st.button(
            "🔍  Analyze Job",
            use_container_width=True
        )

    if analyze_clicked:

        if not job_title.strip() or not job_description.strip():
            st.warning("Please enter at least the Job Title and Job Description.")
        else:
            # IMPORTANT: Reproduce the exact feature pipeline used during training.
            # The training data uses: title + company_profile + description +
            # requirements + benefits. Location is NOT part of the trained text.
            title_clean = clean_text(job_title)
            company_profile_clean = clean_text(company_name)
            description_clean = clean_text(job_description)
            requirements_clean = clean_text(requirements)
            benefits_clean = clean_text(benefits)

            combined_text = " ".join([
                title_clean,
                company_profile_clean,
                description_clean,
                requirements_clean,
                benefits_clean
            ])

            text_vector = tfidf_vectorizer.transform([combined_text])

            # EXACTLY the same 11 engineered features used in main.py.
            description_length = len(description_clean)
            company_profile_length = len(company_profile_clean)
            requirements_length = len(requirements_clean)
            benefits_length = len(benefits_clean)
            suspicious_keyword_score = keyword_score(description_clean)
            missing_company_profile = int(company_profile_clean == "")
            missing_requirements = int(requirements_clean == "")
            missing_benefits = int(benefits_clean == "")

            numeric_features = np.array([[
                description_length,
                company_profile_length,
                requirements_length,
                benefits_length,
                suspicious_keyword_score,
                missing_company_profile,
                missing_requirements,
                missing_benefits,
                int(telecommuting),
                int(has_company_logo),
                int(has_questions)
            ]], dtype=float)

            X_new = hstack([text_vector, numeric_features])

            prediction = int(best_model.predict(X_new)[0])
            decision_score = float(best_model.decision_function(X_new)[0])

            # A simple display confidence based on the SVM decision score.
            confidence = 100 / (1 + np.exp(-abs(decision_score)))

            st.session_state.prediction = prediction
            st.session_state.confidence = confidence
            st.session_state.decision_score = decision_score
            st.session_state.last_company = company_name
            st.session_state.last_job_title = job_title
            st.session_state.last_job_description = job_description
            st.session_state.suspicious_words = get_suspicious_keywords(
                job_title, job_description
            )
            st.session_state.keyword_score = suspicious_keyword_score

            # Save this analysis permanently for the signed-in user.
            if st.session_state.logged_in and st.session_state.user_id:
                save_analysis(
                    st.session_state.user_id,
                    company_name,
                    job_title,
                    "Legitimate" if prediction == 0 else "Fraudulent",
                    confidence
                )

            st.session_state.page = "Result"
            st.rerun()


# ============================================================
# RESULT PAGE
# ============================================================

elif st.session_state.page == "Result":

    prediction = st.session_state.prediction
    confidence = st.session_state.confidence
    job_title = st.session_state.get("last_job_title", "Job Posting")
    job_description = st.session_state.get("last_job_description", "")
    suspicious_words = st.session_state.get(
        "suspicious_words",
        get_suspicious_keywords(job_title, job_description)
    )
    keyword_score_value = st.session_state.get("keyword_score", len(suspicious_words))

    render_html(f"""<div style="text-align:center;padding:55px 20px 30px 20px;">
        <div style="font-size:58px;">{"🟢" if prediction == 0 else "🔴"}</div>
        <div style="font-size:40px;font-weight:900;color:{TEXT};margin-top:12px;">
            {"Likely Legitimate" if prediction == 0 else "Potentially Fraudulent"}
        </div>
        <div style="color:{MUTED};font-size:16px;margin-top:10px;">HireSafe AI assessment for <b>{job_title}</b></div>
    </div>""", unsafe_allow_html=True)

    r1,r2,r3=st.columns(3)
    with r1: st.metric("AI Confidence",f"{confidence:.1f}%")
    with r2: st.metric("Risk Level","Lower Risk" if prediction == 0 else "Higher Risk")
    with r3: st.metric("Result","Legitimate" if prediction == 0 else "Fraudulent")

    if prediction == 0:
        why = (
            "✓ No major suspicious indicators were detected.<br>"
            f"✓ {keyword_score_value} suspicious keyword indicator(s) were found.<br>"
            "✓ The posting matches patterns commonly found in legitimate jobs.<br>"
            "✓ The overall information provided appears reasonably consistent."
        )
        tips=["Verify the employer's official website.","Avoid sharing sensitive information too early.","Confirm the recruiter uses an official company email."]
    else:
        why = (
            "⚠ Suspicious patterns were detected in this job posting.<br>"
            f"⚠ {keyword_score_value} suspicious keyword indicator(s) were found.<br>"
            "⚠ The posting contains characteristics associated with fraudulent jobs.<br>"
            "⚠ Consider verifying the employer before proceeding."
        )
        tips=["Verify the company's official website.","Never pay registration or recruitment fees.","Check company reviews before applying.","Contact the recruiter through official channels."]

    render_html(f"""<div style="background:{CARD};border:1px solid {BORDER};border-radius:18px;padding:24px;margin:25px 0 18px;">
        <div style="font-size:22px;font-weight:800;color:{TEXT};margin-bottom:12px;">🔍 Why this result?</div>
        <div style="color:{MUTED};line-height:2;font-size:15px;">{why}</div>
    </div>""",unsafe_allow_html=True)

    left,right=st.columns(2)
    with left:
        tips_html="".join(f"<div style='margin:9px 0;'>✓ {tip}</div>" for tip in tips)
        render_html(f"""<div style="background:{CARD};border:1px solid {BORDER};border-radius:18px;padding:24px;min-height:190px;">
            <div style="font-size:21px;font-weight:800;color:{TEXT};margin-bottom:12px;">🛡️ Safety Tips</div>
            <div style="color:{MUTED};line-height:1.7;font-size:14px;">{tips_html}</div>
        </div>""",unsafe_allow_html=True)

    with right:
        keyword_html=("".join(f"<div style='margin:9px 0;'>⚠️ {word}</div>" for word in suspicious_words)
            if suspicious_words else "<div style='margin-top:10px;'>No suspicious keywords detected.</div>")
        render_html(f"""<div style="background:{CARD};border:1px solid {BORDER};border-radius:18px;padding:24px;min-height:190px;">
            <div style="font-size:21px;font-weight:800;color:{TEXT};margin-bottom:12px;">🚨 Suspicious Keywords</div>
            <div style="color:{MUTED};line-height:1.7;font-size:14px;">{keyword_html}</div>
        </div>""",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    c1,c2,c3=st.columns([1,2,1])
    with c2:
        if st.button("← Analyze Another Job",use_container_width=True,key="analyze_another_job"):
            st.session_state.page="Analyze"
            st.rerun()


# DASHBOARD PAGE
# ============================================================

if st.session_state.page == "Dashboard":

    if not st.session_state.logged_in:
        render_html(
            f"""
            <div style="text-align:center;padding:80px 20px;">
                <div style="font-size:60px;">👤</div>
                <div style="font-size:40px;font-weight:900;color:{TEXT};">
                    Welcome to your dashboard
                </div>
                <div style="color:{MUTED};font-size:17px;margin-top:12px;">
                    Sign in to keep track of the jobs you have checked.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if st.button("👤 Sign in to continue", use_container_width=True):
                st.session_state.page = "Profile"
                st.rerun()

    else:
        render_html(
            f"""
            <div style="padding:55px 10px 30px 10px;">
                <div style="color:{ACCENT};font-size:14px;font-weight:700;">
                    YOUR DASHBOARD
                </div>
                <div style="font-size:42px;font-weight:900;color:{TEXT};margin-top:8px;">
                    Welcome back, {st.session_state.user_name}
                </div>
                <div style="color:{MUTED};font-size:17px;margin-top:10px;">
                    Here are the job postings you have analyzed.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        user_analyses = get_user_analyses(st.session_state.user_id)

        if not user_analyses:
            render_html(
                f"""
                <div style="
                    background:{CARD};
                    border:1px solid {BORDER};
                    border-radius:20px;
                    padding:50px;
                    text-align:center;
                ">
                    <div style="font-size:50px;">🔎</div>
                    <div style="
                        font-size:24px;
                        font-weight:800;
                        color:{TEXT};
                        margin-top:12px;
                    ">
                        No jobs analyzed yet
                    </div>
                    <div style="
                        color:{MUTED};
                        margin-top:8px;
                    ">
                        Start by checking your first job posting.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                if st.button("🔍 Analyze Your First Job", use_container_width=True):
                    st.session_state.page = "Analyze"
                    st.rerun()

        else:
            for company, job_title, result, confidence, analyzed_at in user_analyses:
                result_color = "#22C55E" if result == "Legitimate" else "#EF4444"
                result_icon = "🟢" if result == "Legitimate" else "🔴"

                try:
                    display_time = datetime.fromisoformat(
                        analyzed_at
                    ).strftime("%d %b %Y, %I:%M %p")
                except ValueError:
                    display_time = analyzed_at

                render_html(
                    f"""
                    <div style="
                        background:{CARD};
                        border:1px solid {BORDER};
                        border-radius:18px;
                        padding:22px;
                        margin-bottom:15px;
                    ">
                        <div style="
                            display:flex;
                            justify-content:space-between;
                            align-items:center;
                            gap:20px;
                        ">
                            <div>
                                <div style="
                                    font-size:20px;
                                    font-weight:800;
                                    color:{TEXT};
                                ">
                                    {company}
                                </div>
                                <div style="
                                    color:{MUTED};
                                    margin-top:5px;
                                ">
                                    {job_title}
                                </div>
                                <div style="
                                    color:{MUTED};
                                    font-size:13px;
                                    margin-top:8px;
                                ">
                                    {display_time}
                                </div>
                            </div>

                            <div style="
                                color:{result_color};
                                font-weight:800;
                                white-space:nowrap;
                            ">
                                {result_icon} {result}
                                <div style="
                                    color:{MUTED};
                                    font-size:12px;
                                    text-align:right;
                                    margin-top:5px;
                                ">
                                    {confidence:.1f}% confidence
                                </div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ABOUT PAGE
# ============================================================

elif st.session_state.page == "About":

    render_html(
        f"""
        <div style="text-align:center;padding:65px 20px 35px 20px;">
            <div style="font-size:55px;">🛡️</div>
            <div style="
                font-size:44px;
                font-weight:900;
                color:{TEXT};
                margin-top:12px;
            ">
                Why HireSafe AI?
            </div>
            <div style="
                max-width:720px;
                margin:15px auto 0;
                color:{MUTED};
                font-size:17px;
                line-height:1.7;
            ">
                Looking for a job should be exciting, not risky.
                HireSafe AI was created to help job seekers pause,
                check the opportunity, and make a more informed decision
                before sharing their information or applying.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    a1, a2, a3 = st.columns(3)

    with a1:
        render_html(
            f"""
            <div class="feature">
                <div class="feature-icon">🔍</div>
                <div class="feature-title">Check Before You Apply</div>
                <div class="feature-text">
                    Review a posting and identify warning signs before taking the next step.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with a2:
        render_html(
            f"""
            <div class="feature">
                <div class="feature-icon">🛡️</div>
                <div class="feature-title">Reduce Unnecessary Risk</div>
                <div class="feature-text">
                    Make it easier to recognize postings that deserve a closer look.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with a3:
        render_html(
            f"""
            <div class="feature">
                <div class="feature-icon">💼</div>
                <div class="feature-title">Apply With Confidence</div>
                <div class="feature-text">
                    Get a clear result that helps you decide what to do next.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br><br>", unsafe_allow_html=True)

    render_html(
        f"""
        <div style="
            max-width:850px;
            margin:auto;
            background:{CARD};
            border:1px solid {BORDER};
            border-radius:20px;
            padding:30px;
        ">
            <div style="
                color:{ACCENT};
                font-size:13px;
                font-weight:800;
                letter-spacing:1px;
            ">
                OUR MOTIVE
            </div>

            <div style="
                color:{TEXT};
                font-size:24px;
                font-weight:800;
                margin-top:10px;
            ">
                A safer job search starts with one simple check.
            </div>

            <div style="
                color:{MUTED};
                font-size:16px;
                line-height:1.8;
                margin-top:12px;
            ">
                Fake job postings can waste time, collect personal information,
                or create financial risks for job seekers. HireSafe AI aims to
                give candidates a simple first layer of awareness before they
                move forward with an opportunity.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# PROFILE FALLBACK PAGE
# ============================================================

elif st.session_state.page == "Profile":

    render_html(
        f"""
        <div style="text-align:center;padding:55px 20px 25px;">
            <div style="font-size:50px;">👤</div>
            <div style="font-size:38px;font-weight:900;color:{TEXT};">
                {"Welcome, " + st.session_state.user_name if st.session_state.logged_in else "Welcome to HireSafe AI"}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.logged_in:

        st.success(
            f"Signed in as {st.session_state.user_name}"
        )

        if st.button(
            "Sign out",
            key="profile_page_signout"
        ):
            st.session_state.logged_in = False
            st.session_state.user_name = ""
            st.session_state.user_email = ""
            st.session_state.user_id = None
            st.session_state.page = "Home"
            st.rerun()

    else:

        sign_in_tab, sign_up_tab = st.tabs(
            ["Sign In", "Sign Up"]
        )

        with sign_in_tab:

            email = st.text_input(
                "Email",
                key="profile_login_email"
            )

            password = st.text_input(
                "Password",
                type="password",
                key="profile_login_password"
            )

            if st.button(
                "Sign In",
                key="profile_signin"
            ):

                user = authenticate_user(
                    email,
                    password
                )

                if user:
                    st.session_state.logged_in = True
                    st.session_state.user_id = user["id"]
                    st.session_state.user_name = user["name"]
                    st.session_state.user_email = user["email"]
                    st.session_state.page = "Dashboard"
                    st.rerun()

                else:
                    st.error(
                        "Invalid email or password."
                    )

        with sign_up_tab:

            name = st.text_input(
                "Name",
                key="profile_signup_name"
            )

            email = st.text_input(
                "Email",
                key="profile_signup_email"
            )

            password = st.text_input(
                "Password",
                type="password",
                key="profile_signup_password"
            )

            confirm = st.text_input(
                "Confirm Password",
                type="password",
                key="profile_signup_confirm"
            )

            if st.button(
                "Create Account",
                key="profile_signup"
            ):

                if not all([
                    name.strip(),
                    email.strip(),
                    password,
                    confirm
                ]):
                    st.warning(
                        "Please fill in all fields."
                    )

                elif password != confirm:
                    st.error(
                        "Passwords do not match."
                    )

                elif len(password) < 6:
                    st.warning(
                        "Password must be at least 6 characters."
                    )

                else:
                    user_id, error = create_user(
                        name,
                        email,
                        password
                    )

                    if error:
                        st.error(error)

                    else:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user_id
                        st.session_state.user_name = name.strip()
                        st.session_state.user_email = email.strip().lower()
                        st.session_state.page = "Dashboard"
                        st.rerun()


# MAIN HERO
# ============================================================

if st.session_state.page == "Home":

    render_html(
        f"""
    <div class="hero-container">

        <div class="hero-shield">
            🛡️
        </div>

        <div class="hero-title">
            HIRE<span>SAFE</span> AI
        </div>

        <div class="hero-subtitle">
            AI-Powered Fake Job Detection
        </div>

        <div class="hero-description">
            Looking for a job should be exciting, not risky. 
            HireSafe AI helps you check job postings for warning signs so you can apply with confidence.
        </div>

    </div>
    """,
        unsafe_allow_html=True
    )


    # ============================================================
    # START ANALYSIS
    # ============================================================

    c1, c2, c3 = st.columns([1, 1, 1])

    with c2:

        if st.button(
            "🔍  Start Job Analysis",
            use_container_width=True
        ):
            st.session_state.page = "Analyze"
            st.rerun()


    # ============================================================
    # FEATURES
    # ============================================================

    st.markdown("<br>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)


    with f1:

        render_html(
            f"""
    <div class="feature">

        <div class="feature-icon">
            🤖
        </div>

        <div class="feature-title">
            AI Detection
        </div>

        <div class="feature-text">
            Machine learning analyzes job postings
            for fraudulent patterns.
        </div>

    </div>
    """,
            unsafe_allow_html=True
        )


    with f2:

        render_html(
            f"""
    <div class="feature">

        <div class="feature-icon">
            🧠
        </div>

        <div class="feature-title">
            NLP Analysis
        </div>

        <div class="feature-text">
            TF-IDF and Natural Language Processing
            analyze the content of job postings.
        </div>

    </div>
    """,
            unsafe_allow_html=True
        )


    with f3:

        render_html(
            f"""
    <div class="feature">

        <div class="feature-icon">
            🛡️
        </div>

        <div class="feature-title">
            Stay Safe
        </div>

        <div class="feature-text">
            Detect suspicious job postings before
            sharing personal information.
        </div>

    </div>
    """,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

if st.session_state.page == "Home":

    render_html(
        f"""
    <div style="
        text-align:center;
        margin-top:80px;
        padding-top:25px;
        border-top:1px solid {BORDER};
        color:{MUTED};
        font-size:13px;
    ">
        HireSafe AI • Machine Learning • NLP • Explainable AI
    </div>
    """,
        unsafe_allow_html=True
    )