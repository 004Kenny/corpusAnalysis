

from __future__ import annotations
import io
import time
import re
import sys
from typing import Iterable, Tuple, Dict, List
from urllib.parse import urlparse
from pathlib import Path
import tempfile
import logging

import requests
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup

# PDF extraction
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pdfminer.pdfpage import PDFPage
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# NLP
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ------------------------- Robust NLTK setup with fallback -------------------------
USE_NLTK = True
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        USE_NLTK = False
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    try:
        nltk.download("stopwords", quiet=True)
    except Exception:
        USE_NLTK = False

FALLBACK_STOPWORDS = {
    "the","and","to","of","in","a","is","it","that","for","on","as","with","this","by","an","are","be",
    "or","from","at","was","but","not","have","has","had","were","which","their","its","they","we","you",
    "your","our","can","will","would","should","could","about","into","over","than","so","no","yes","if",
    "when","while","what","who","whom","where","why","how","all","any","each","few","more","most","other",
    "some","such","only","own","same","both","very","s","t","just","don","now"
}

if USE_NLTK:
    try:
        STOPWORDS = set(stopwords.words("english"))
    except Exception:
        STOPWORDS = FALLBACK_STOPWORDS
        USE_NLTK = False
else:
    STOPWORDS = FALLBACK_STOPWORDS

if USE_NLTK:
    from nltk import sent_tokenize, word_tokenize
else:
    import re as _re
    def sent_tokenize(text: str) -> List[str]:
        return [s.strip() for s in _re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    def word_tokenize(text: str) -> List[str]:
        return _re.findall(r"[A-Za-z']+", text)

# ------------------------- Helpers -------------------------
UA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

def is_pdf_link(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")

def fetch_bytes(url: str, timeout: int = 60) -> Tuple[bytes, str]:
    r = requests.get(url, headers=UA_HEADERS, timeout=timeout)
    r.raise_for_status()
    ctype = (r.headers.get("Content-Type") or "").lower()
    return r.content, ctype

# HTML extraction
COMMON_WRAPPERS = ("article", "main", "#content", ".content", ".entry-content", ".article-body")

def extract_html_sections(html: str, body_selector: str | None = None) -> Iterable[Tuple[int, str]]:
    soup = BeautifulSoup(html, "lxml")
    if body_selector:
        nodes = soup.select(body_selector)
        if nodes:
            text = " ".join(n.get_text(" ", strip=True) for n in nodes)
            if text.strip():
                yield 0, text
                return
    # heuristics
    article = soup.find("article")
    if article:
        text = " ".join(p.get_text(" ", strip=True) for p in article.find_all(["p", "li"]))
        if text.strip():
            yield 0, text
            return
    for sel in COMMON_WRAPPERS:
        block = soup.select_one(sel)
        if block:
            text = " ".join(p.get_text(" ", strip=True) for p in block.find_all(["p", "li"]))
            if text.strip():
                yield 0, text
                return
    # fallback: all paragraphs
    ps = soup.find_all("p")
    text = " ".join(p.get_text(" ", strip=True) for p in ps)
    if text.strip():
        yield 0, text

# PDF extraction

def extract_pdf_pages_to_text(pdf_path: Path) -> Iterable[Tuple[int, str]]:
    # Count pages
    with open(pdf_path, "rb") as f:
        num_pages = sum(1 for _ in PDFPage.get_pages(f))
    for i in range(num_pages):
        try:
            page_text = pdfminer_extract_text(str(pdf_path), page_numbers=[i]) or ""
        except Exception as e:
            page_text = ""
        yield i, page_text

# NLP pipeline per text

def analyze_text(text: str, top_k: int = 25) -> Dict:
    if not text:
        return {"sentences": [], "tokens": [], "clean": [], "top": []}
    sentences = sent_tokenize(text)
    raw_tokens = word_tokenize(text.lower())
    tokens = [t for t in raw_tokens if re.fullmatch(r"[a-z]+", t) and len(t) > 2]
    clean = [t for t in tokens if t not in STOPWORDS]
    # frequency
    # Manual frequency to avoid importing NLTK FreqDist when in fallback
    freq: Dict[str, int] = {}
    for t in clean:
        freq[t] = freq.get(t, 0) + 1
    top = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return {"sentences": sentences, "tokens": tokens, "clean": clean, "top": top}

# WordCloud

def render_wordcloud(tokens: List[str]):
    if not tokens:
        return None
    text = " ".join(tokens)
    cloud = WordCloud(width=1600, height=900, background_color="white", random_state=42).generate(text)
    fig = plt.figure(figsize=(9, 5))
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    return fig

# ------------------------- Streamlit UI -------------------------
st.set_page_config(page_title="Ebook Scraper + NLP", layout="wide")
st.title("📘 Ebook Scraper + NLP (PDF/HTML)")

with st.sidebar:
    st.header("Settings")
    default_url = st.session_state.get("last_url", "")
    url = st.text_input("Ebook URL", value=default_url, placeholder="https://... .pdf or .html")

# Defaults (no sidebar controls)
body_selector = None
make_clouds = True
per_page_clouds = True
cloud_interval = 10.0
show_top_chart = True
top_k = 25
max_pages = 0
max_sections = 0

# Main action button (moved out of sidebar)
go = st.button("Scrape & Analyze", type="primary")

left, right = st.columns([2, 1])

if go:
    if not url.strip():
        st.error("Please enter a valid URL.")
        st.stop()

    st.session_state["last_url"] = url

    with st.spinner("Fetching URL..."):
        try:
            content_bytes, ctype = fetch_bytes(url)
        except Exception as e:
            st.error(f"Fetch failed: {e}")
            st.stop()

    is_pdf = "pdf" in ctype or is_pdf_link(url)

    # Prepare outputs
    rows_summary: List[Dict] = []
    aggregate_tokens: List[str] = []

    wordcloud_placeholder = left.empty()
    progress = left.progress(0)
    status = left.empty()

    last_cloud_time = 0.0

    if is_pdf:
        # Save bytes to temp and iterate pages
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content_bytes)
            pdf_path = Path(tmp.name)
        try:
            # Count pages for progress
            with open(pdf_path, "rb") as f:
                total_pages = sum(1 for _ in PDFPage.get_pages(f))
            if max_pages and max_pages > 0:
                total_iter = min(total_pages, int(max_pages))
            else:
                total_iter = total_pages

            for i, page_text in extract_pdf_pages_to_text(pdf_path):
                if max_pages and max_pages > 0 and i >= max_pages:
                    break
                res = analyze_text(page_text, top_k=int(top_k))
                aggregate_tokens.extend(res["clean"])
                rows_summary.append({
                    "index": i,
                    "n_sentences": len(res["sentences"]),
                    "n_tokens": len(res["tokens"]),
                    "n_clean_tokens": len(res["clean"]),
                    "top_words": " ".join([f"{w}:{c}" for w, c in res["top"]]),
                })

                # WordCloud per page (throttled)
                if make_clouds and per_page_clouds:
                    now = time.time()
                    if now - last_cloud_time >= float(cloud_interval):
                        fig = render_wordcloud(res["clean"])
                        if fig is not None:
                            wordcloud_placeholder.pyplot(fig)
                        last_cloud_time = now

                progress.progress(min((i + 1) / max(total_iter, 1), 1.0))
                status.write(f"Processed page {i+1}/{total_pages}")

        finally:
            try:
                pdf_path.unlink(missing_ok=True)
            except Exception:
                pass

    else:
        # HTML: decode and extract sections (usually 1 section)
        try:
            html = content_bytes.decode("utf-8", "ignore")
        except Exception:
            html = content_bytes.decode(errors="ignore")
        sec_iter = list(extract_html_sections(html, body_selector if (body_selector and body_selector.strip()) else None))
        if max_sections and max_sections > 0:
            sec_iter = sec_iter[: int(max_sections)]
        total_sections = len(sec_iter)
        for idx, sec_text in sec_iter:
            res = analyze_text(sec_text, top_k=int(top_k))
            aggregate_tokens.extend(res["clean"])
            rows_summary.append({
                "index": idx,
                "n_sentences": len(res["sentences"]),
                "n_tokens": len(res["tokens"]),
                "n_clean_tokens": len(res["clean"]),
                "top_words": " ".join([f"{w}:{c}" for w, c in res["top"]]),
            })

            # One cloud per section if throttled interval passed
            if make_clouds and per_page_clouds:
                now = time.time()
                if now - last_cloud_time >= float(cloud_interval):
                    fig = render_wordcloud(res["clean"])
                    if fig is not None:
                        wordcloud_placeholder.pyplot(fig)
                    last_cloud_time = now

            progress.progress(min((idx + 1) / max(total_sections, 1), 1.0))
            status.write(f"Processed section {idx+1}/{total_sections}")

    # Aggregate results

    # Top words on aggregate
    agg_res = analyze_text(" ".join(aggregate_tokens)) if aggregate_tokens else {"top": []}

    # Summary table and download
    df_summary = pd.DataFrame(rows_summary)
    if not df_summary.empty:
        left.subheader("Sections Summary")
        left.dataframe(df_summary, use_container_width=True, hide_index=True)
        csv_buf = io.StringIO()
        df_summary.to_csv(csv_buf, index=False)
        st.download_button(
            label="Download sections summary CSV",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name="ebook_sections_summary.csv",
            mime="text/csv",
        )

    # Show top words bar chart
    top_pairs = agg_res.get("top", [])
    if show_top_chart and top_pairs:
        right.subheader("Top Words (Aggregate)")
        df_top = pd.DataFrame(top_pairs, columns=["word", "count"])
        right.bar_chart(df_top.set_index("word"))

    # Final aggregate cloud
    if make_clouds:
        fig = render_wordcloud(aggregate_tokens)
        if fig is not None:
            right.subheader("Aggregate WordCloud")
            right.pyplot(fig)

    st.success("Done.")
