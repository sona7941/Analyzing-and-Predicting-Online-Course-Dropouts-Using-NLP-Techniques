# dropout_prediction_nlp.py
"""
Streamlit app — End-to-end NLP-based Student Dropout Analysis & Prediction

Features:
- Upload CSV of historical student feedback (expects columns like: feedback, status, subject, country)
- Clean & enrich text using NLTK + spaCy (POS, simple NER counts)
- Vectorize with TF-IDF (+ optional Word2Vec if gensim available)
- Train/test split + Logistic Regression
- Evaluation: accuracy, confusion matrix, classification report
- Visuals: which subject and which country have maximum dropouts (counts & rates)
- Interactive predictions:
    1) Quick text box to predict one feedback
    2) A structured feedback form (name, country, subject, feedback) with on-the-fly prediction
- Auto-generated Markdown report for download

Notes:
- Word2Vec (gensim) is optional. If gensim isn't installed, the app will run without it.
- spaCy model `en_core_web_sm` is auto-downloaded if missing (requires internet the first time).
- The app is robust to alternate column names (e.g., "course" instead of "subject", "location" instead of "country").

Author: (you)
"""

# =====================
# Imports & Setup
# =====================
import os
import io
import re
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Tuple

# Sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)
from sklearn.metrics.pairwise import cosine_similarity

# NLTK
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# spaCy (auto-download model if missing)
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    with st.spinner("Downloading spaCy model en_core_web_sm (first-run only)..."):
        from spacy.cli import download as spacy_download
        spacy_download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# gensim (optional)
W2V_AVAILABLE = True
try:
    from gensim.models import Word2Vec
except Exception:
    W2V_AVAILABLE = False

# =====================
# Streamlit Page Config
# =====================
st.set_page_config(page_title="NLP Dropout Predictor", page_icon="📊", layout="wide")
st.title("📊 NLP-based Student Dropout Prediction — Full App")
st.write(
    "This app showcases **NLP + ML** for dropout risk using student feedback, plus data exploration and reporting."
)

# =====================
# Helper Functions
# =====================

def get_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """Return the first column name from candidates that exists in df (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def preprocess_text(text: str) -> str:
    """Basic text cleaning + tokenization + stopword removal."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    stop = set(stopwords.words("english"))
    tokens = [t for t in tokens if t and t not in stop]
    return " ".join(tokens)


def pos_features(text: str) -> pd.Series:
    """Count POS tags: NOUN, VERB, ADJ, ADV using spaCy."""
    doc = nlp(text)
    counts = {"NOUN": 0, "VERB": 0, "ADJ": 0, "ADV": 0}
    for token in doc:
        if token.pos_ in counts:
            counts[token.pos_] += 1
    return pd.Series(counts)


def ner_counts(text: str) -> pd.Series:
    """Count simple entity labels: GPE, DATE, ORG using spaCy NER."""
    doc = nlp(text)
    ents = {"GPE": 0, "DATE": 0, "ORG": 0}
    for ent in doc.ents:
        if ent.label_ in ents:
            ents[ent.label_] += 1
    return pd.Series(ents)


def build_features(df: pd.DataFrame, feedback_col: str) -> Tuple[np.ndarray, dict]:
    """Builds TF-IDF, optional Word2Vec, POS/NER counts, cosine sim proto features.

    Returns:
        X: feature matrix
        context: a dict with fitted objects (tfidf, w2v_model, feature_names, etc.)
    """
    clean_col = f"clean__{feedback_col}"
    if clean_col not in df:
        df[clean_col] = df[feedback_col].apply(preprocess_text)

    # POS & NER feature frames
    pos_df = df[clean_col].apply(pos_features)
    ner_df = df[clean_col].apply(ner_counts)

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=400)
    tfidf_features = tfidf.fit_transform(df[clean_col]).toarray()

    # Word2Vec (optional)
    w2v_features = None
    w2v_model = None
    if W2V_AVAILABLE:
        tokenized = [t.split() for t in df[clean_col]]
        w2v_model = Word2Vec(sentences=tokenized, vector_size=50, window=5, min_count=1, workers=4)

        def get_w2v_vector(tokens):
            vecs = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
            if not vecs:
                return np.zeros(50)
            return np.mean(vecs, axis=0)

        w2v_features = np.vstack([get_w2v_vector(t.split()) for t in df[clean_col]])

    # Cosine similarity to prototypical phrases (via TF-IDF space)
    positive_proto = "helpful excellent great satisfied"
    negative_proto = "boring bad difficult disappointed"
    pos_vec = tfidf.transform([positive_proto]).toarray()
    neg_vec = tfidf.transform([negative_proto]).toarray()
    feedback_vecs = tfidf.transform(df[clean_col]).toarray()
    cos_pos = cosine_similarity(feedback_vecs, pos_vec).flatten()
    cos_neg = cosine_similarity(feedback_vecs, neg_vec).flatten()

    # Assemble final X
    extra = pd.concat([pos_df, ner_df], axis=1)
    base_blocks = [tfidf_features]
    if w2v_features is not None:
        base_blocks.append(w2v_features)
    base_blocks.append(extra.values)
    base_blocks.append(np.c_[cos_pos, cos_neg])
    X = np.hstack(base_blocks)

    context = {
        "clean_col": clean_col,
        "tfidf": tfidf,
        "w2v_model": w2v_model,
        "use_w2v": w2v_features is not None,
        "pos_proto": pos_vec,
        "neg_proto": neg_vec,
        "feature_blocks": {
            "tfidf": tfidf_features.shape[1],
            "w2v": 50 if w2v_features is not None else 0,
            "pos_ner": extra.shape[1],
            "cos": 2,
        },
    }
    return X, context


def vectorize_one(text: str, ctx: dict) -> np.ndarray:
    """Vectorize a single raw text using fitted context."""
    clean = preprocess_text(text)
    pos_s = pos_features(clean)
    ner_s = ner_counts(clean)

    tfidf_vec = ctx["tfidf"].transform([clean]).toarray()

    w2v_vec = None
    if ctx.get("use_w2v") and ctx.get("w2v_model") is not None:
        tokens = clean.split()
        w2v_model = ctx["w2v_model"]
        vecs = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
        w2v_vec = np.mean(vecs, axis=0) if vecs else np.zeros(50)
        w2v_vec = w2v_vec.reshape(1, -1)

    cos_pos_u = cosine_similarity(tfidf_vec, ctx["pos_proto"]).flatten()[0]
    cos_neg_u = cosine_similarity(tfidf_vec, ctx["neg_proto"]).flatten()[0]

    blocks = [tfidf_vec]
    if w2v_vec is not None:
        blocks.append(w2v_vec)
    blocks.append(pos_s.values.reshape(1, -1))
    blocks.append(ner_s.values.reshape(1, -1))
    blocks.append(np.array([[cos_pos_u, cos_neg_u]]))

    return np.hstack(blocks)


# =====================
# Sidebar — CSV Upload & Options
# =====================
with st.sidebar:
    st.header("📁 Data Upload")
    file = st.file_uploader("Upload CSV with columns (feedback, status, subject, country)", type=["csv"])        
    st.caption("Status values should be 'Dropout' or 'Continue' (case-insensitive). Other values will be coerced.")

    st.header("⚙️ Options")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", value=42, step=1)

# Early exit if no file
if not file:
    st.info("👆 Upload a CSV in the sidebar to begin. Example columns: feedback, status, subject, country.")
    st.stop()

# =====================
# Load Data & Standardize Columns
# =====================
df = pd.read_csv(file)

# Flexible column detection
feedback_col = get_column(df, ["feedback", "comment", "review", "text"])
status_col   = get_column(df, ["status", "label", "target", "dropout"])
subject_col  = get_column(df, ["subject", "course", "module", "class"])
country_col  = get_column(df, ["country", "location", "nation"])

missing = [name for name, col in {
    "feedback": feedback_col,
    "status": status_col,
}.items() if col is None]

if missing:
    st.error(f"Missing required columns: {', '.join(missing)}. Please include at least feedback & status columns.")
    st.stop()

# Coerce status to {Dropout, Continue}
df[status_col] = df[status_col].astype(str).str.strip().str.lower().map({
    "dropout": "Dropout",
    "dropped": "Dropout",
    "1": "Dropout",
    "continue": "Continue",
    "completed": "Continue",
    "0": "Continue",
}).fillna(df[status_col])

# Show sample
st.subheader("👀 Sample Data")
st.dataframe(df.head(), use_container_width=True)

# =====================
# Feature Engineering & Model Training
# =====================
X, ctx = build_features(df, feedback_col)
y = (df[status_col].astype(str).str.lower() == "dropout").astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_state), stratify=y
)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Eval
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

col1, col2 = st.columns([1, 1])
with col1:
    st.metric("✅ Accuracy", f"{acc:.3f}")
with col2:
    st.write("**Model Info**")
    st.write({k: v for k, v in ctx["feature_blocks"].items()})
    st.caption("Numbers show feature block sizes used in training.")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm,
            xticklabels=["Continue", "Dropout"], yticklabels=["Continue", "Dropout"])
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
ax_cm.set_title("Confusion Matrix")
st.pyplot(fig_cm)

# Classification report
report_str = classification_report(y_test, y_pred, target_names=["Continue", "Dropout"])
st.text("Classification Report:\n" + report_str)

# =====================
# Exploratory Charts: Subjects & Countries
# =====================
st.subheader("📈 Exploratory Insights")

if subject_col is not None:
    left, right = st.columns(2)
    with left:
        st.write("**Dropout Counts by Subject**")
        subj_counts = df[df[status_col] == "Dropout"][subject_col].value_counts().sort_values(ascending=False)
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        subj_counts.head(20).plot(kind="bar", ax=ax1)
        ax1.set_xlabel("Subject")
        ax1.set_ylabel("# Dropouts")
        ax1.set_title("Top Subjects by Dropout Count")
        st.pyplot(fig1)
        if len(subj_counts) > 0:
            st.success(f"Subject with max dropouts: **{subj_counts.index[0]}** ({subj_counts.iloc[0]})")

    with right:
        st.write("**Dropout Rate by Subject**")
        rate_df = (
            df.assign(y=(df[status_col] == "Dropout").astype(int))
              .groupby(subject_col)['y']
              .agg(['count', 'sum'])
              .rename(columns={'count': 'n', 'sum': 'dropouts'})
        )
        rate_df['rate'] = rate_df['dropouts'] / rate_df['n']
        rate_df = rate_df.sort_values('rate', ascending=False)
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        rate_df['rate'].head(20).plot(kind="bar", ax=ax2)
        ax2.set_xlabel("Subject")
        ax2.set_ylabel("Dropout Rate")
        ax2.set_title("Top Subjects by Dropout Rate")
        st.pyplot(fig2)
else:
    st.info("No subject/course column found; skipping subject-level charts.")

if country_col is not None:
    left, right = st.columns(2)
    with left:
        st.write("**Dropout Counts by Country**")
        c_counts = df[df[status_col] == "Dropout"][country_col].value_counts().sort_values(ascending=False)
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        c_counts.head(20).plot(kind="bar", ax=ax3)
        ax3.set_xlabel("Country")
        ax3.set_ylabel("# Dropouts")
        ax3.set_title("Top Countries by Dropout Count")
        st.pyplot(fig3)
        if len(c_counts) > 0:
            st.success(f"Country with max dropouts: **{c_counts.index[0]}** ({c_counts.iloc[0]})")

    with right:
        st.write("**Dropout Rate by Country**")
        rate_c = (
            df.assign(y=(df[status_col] == "Dropout").astype(int))
              .groupby(country_col)['y']
              .agg(['count', 'sum'])
              .rename(columns={'count': 'n', 'sum': 'dropouts'})
        )
        rate_c['rate'] = rate_c['dropouts'] / rate_c['n']
        rate_c = rate_c.sort_values('rate', ascending=False)
        fig4, ax4 = plt.subplots(figsize=(5, 3))
        rate_c['rate'].head(20).plot(kind="bar", ax=ax4)
        ax4.set_xlabel("Country")
        ax4.set_ylabel("Dropout Rate")
        ax4.set_title("Top Countries by Dropout Rate")
        st.pyplot(fig4)
else:
    st.info("No country/location column found; skipping country-level charts.")

# =====================
# Interactive Predictions
# =====================
st.subheader("🧪 Try Predictions")

# Quick single-text prediction
user_text = st.text_area("Enter a feedback to predict dropout risk:")
if user_text.strip():
    x_one = vectorize_one(user_text, ctx)
    pred = model.predict(x_one)[0]
    proba = model.predict_proba(x_one)[0][pred]
    if pred == 1:
        st.error(f"🔴 Predicted: **Dropout** (p={proba:.2f})")
    else:
        st.success(f"🟢 Predicted: **Continue** (p={proba:.2f})")

st.divider()

# Structured feedback form
st.subheader("📝 Feedback Form (with Prediction)")
with st.form("feedback_form"):
    colA, colB, colC = st.columns(3)
    with colA:
        name = st.text_input("Student Name", "")
    with colB:
        subj_val = st.text_input("Subject/Course", "") if subject_col is None else st.selectbox(
            "Subject/Course", sorted(df[subject_col].dropna().unique().tolist())
        )
    with colC:
        country_val = st.text_input("Country", "") if country_col is None else st.selectbox(
            "Country", sorted(df[country_col].dropna().unique().tolist())
        )

    feedback_val = st.text_area("Feedback", "")
    submitted = st.form_submit_button("Predict from Feedback")

if submitted and feedback_val.strip():
    x_form = vectorize_one(feedback_val, ctx)
    pred = model.predict(x_form)[0]
    proba = model.predict_proba(x_form)[0][pred]
    msg = f"**Prediction:** {'Dropout' if pred == 1 else 'Continue'} (p={proba:.2f})\n\n"
    msg += f"**Name:** {name or '-'}  |  **Subject:** {subj_val or '-'}  |  **Country:** {country_val or '-'}"
    if pred == 1:
        st.error(msg)
    else:
        st.success(msg)

# =====================
# Downloadable Report (Markdown)
# =====================
st.subheader("📄 Generate Report")

def build_markdown_report() -> str:
    lines = []
    lines.append(f"# Dropout Analysis Report")
    lines.append("")
    lines.append(f"**Records:** {len(df)}  ")
    lines.append(f"**Accuracy:** {acc:.3f}")
    lines.append("")
    lines.append("## Confusion Matrix")
    lines.append("Columns/rows: [Continue, Dropout]")
    lines.append("\n" + pd.DataFrame(cm, index=["Actual-Cont", "Actual-Drop"], columns=["Pred-Cont", "Pred-Drop"]).to_markdown())
    lines.append("")
    lines.append("## Classification Report")
    lines.append("```")
    lines.append(report_str)
    lines.append("```")

    # Subject summary
    if subject_col is not None:
        lines.append("## Subjects with Highest Dropouts")
        top_subj = df[df[status_col] == "Dropout"][subject_col].value_counts().head(10)
        if len(top_subj) > 0:
            lines.append(top_subj.to_markdown())
        else:
            lines.append("No dropout records by subject.")

    # Country summary
    if country_col is not None:
        lines.append("\n## Countries with Highest Dropouts")
        top_c = df[df[status_col] == "Dropout"][country_col].value_counts().head(10)
        if len(top_c) > 0:
            lines.append(top_c.to_markdown())
        else:
            lines.append("No dropout records by country.")

    lines.append("")
    lines.append("---")
    lines.append("_Report generated by the Streamlit app._")
    return "\n".join(lines)

report_md = build_markdown_report()
report_bytes = report_md.encode("utf-8")
st.download_button(
    label="⬇️ Download Markdown Report",
    data=report_bytes,
    file_name="dropout_report.md",
    mime="text/markdown",
)

st.caption("Tip: Convert the Markdown to PDF using any Markdown-to-PDF tool if needed.")

# =====================
# Footer
# =====================
st.info(
    "This app extracts interpretable text features (POS/NER/TF‑IDF, optional Word2Vec) and uses Logistic Regression. "
    "You can replace the classifier with SVM/RandomForest, or swap TF‑IDF for pretrained embeddings as an extension."
)
