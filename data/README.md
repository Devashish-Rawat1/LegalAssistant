# Data Sourcing Guide

This folder holds the raw statute text that `ingest.py` cleans, chunks, and indexes.

## Where to get the acts (Week 1 task — Akshit)

**Primary source: India Code — https://www.indiacode.nic.in**
Official, government-maintained repository of every central act. This is the
authoritative source and should be preferred over any third-party copy.

Steps:
1. Go to India Code → search the act by name (e.g. "Bharatiya Nyaya Sanhita 2023").
2. Open the act page and download the official PDF.
3. Extract text from the PDF (see below).
4. Save the cleaned result as `data/<act_slug>.txt` (e.g. `data/bns_2023.txt`).

**Cross-check source: Legislative Department — https://legislative.gov.in**
Useful for verifying gazette-notified text, especially for the newer 2023
criminal law acts (BNS, BNSS, BSA), since these are recent enough that some
third-party copies online are outdated or incomplete.

**Faster bootstrap (verify before trusting): Kaggle / HuggingFace datasets**
Search "Indian Penal Code dataset" or "Indian legal corpus" for pre-structured,
section-split text. Useful to move quickly, but cross-check a sample of
sections against India Code before relying on it — crowd-sourced legal text
sometimes has OCR errors or misses amendments.

## Extracting text from PDFs

```bash
pip install pdfplumber
```

```python
import pdfplumber

with pdfplumber.open("bns_2023.pdf") as pdf:
    text = "\n".join(page.extract_text() or "" for page in pdf.pages)

with open("data/bns_2023.txt", "w", encoding="utf-8") as f:
    f.write(text)
```

Expect to manually fix a few OCR artifacts (broken line breaks, stray page
numbers/headers) — `ingest.py`'s `clean_text()` handles basic normalization,
but section-numbering irregularities are easiest to fix by hand on a first pass.

## File naming convention

One `.txt` file per act, lowercase, underscore-separated:

```
data/
├── constitution_of_india.txt
├── bns_2023.txt
├── bnss_2023.txt
├── bsa_2023.txt
├── consumer_protection_act_2019.txt
├── motor_vehicles_act_1988.txt
└── it_act_2000.txt
```

`ingest.py` picks up every `.txt` file in this folder automatically and tags
each chunk with its filename as the source act — this is what will later
power section/act citation in answers.
