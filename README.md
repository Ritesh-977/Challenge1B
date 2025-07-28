# 🧠 Intelligent PDF Section Extractor

A lightweight, offline tool that intelligently extracts the **most relevant sections** from a collection of PDFs based on a given **persona** and **job-to-be-done**—ideal for analysts, auditors, reviewers, and decision-makers.


## 🚀 What It Does

Given:
- A user’s **role** (persona)
- A **task** to accomplish
- A set of PDF documents

This tool:
1. Detects **headings** based on PDF font size and structure
2. Splits content into **sections** with titles, body text, and page numbers
3. Computes **semantic similarity** using TF-IDF and cosine similarity
4. Ranks and extracts the **top N relevant sections**
5. Identifies the **most relevant paragraph** within each section

---

## Models & Libraries Used

| Library        | Role |
|----------------|------|
| `pdfplumber`   | PDF parsing with font-size & layout metadata |
| `scikit-learn` | TF-IDF vectorization & cosine similarity scoring |
| `numpy`        | Numerical ops |
| `re`, `json`   | Paragraph splitting, structured I/O |

> ✅ No deep learning models used — fully CPU-based, lightweight, and runs offline
> ✅ Optimized for PDFs with up to ~50 pages and basic formatting


## 🛠️ How to Build & Run

### Step 1: Build Docker Image

docker build --no-cache --platform linux/amd64 -t myapp .

### Step 2: Run the Container
 docker run --rm -v "${PWD}:/app" myapp  

> 🔒 Runs completely offline with `--network none`  


## ✅ Highlights

- ⚡ **Fast**: Processes 3–10 PDFs in under 60 seconds
- 🧠 **Smart**: Automatically detects section boundaries using real font sizes
- 🔍 **Relevant**: Uses semantic ranking to find meaningful content, not just keywords
- 🌐 **Offline**: Fully Dockerized and air-gapped — no internet or cloud required
- 🧩 **Hackathon-Ready**: Easy to test, deploy, and extend

