# Financial-event-clustering and RAG implementation

> **Detecting and analyzing financial market events from news using NLP pipelines — NASDAQ, Jan–Mar 2026.**

This project builds an end-to-end NLP system that automatically clusters financial news articles into coherent groups corresponding to real-world market events. Two embedding paradigms are compared on real NASDAQ data, and a local RAG system enables natural language querying of the entire document base.

---

## Overview

```
Raw Data (GDELT + Telegram + Yahoo Finance)
        ↓
  Lexicon Generation (Marginal Screening f(j))
        ↓
  Feature Engineering (Word2Vec / DistilRoBERTa)
        ↓
  Clustering (HAC + Silhouette optimization)
        ↓
  Local RAG (DistilRoBERTa retrieval + Qwen 2.5 0.5B generation)
```

The core idea: build a **time-aware financial lexicon** calibrated on actual NASDAQ daily returns, use it to filter and represent news articles as dense vectors, then cluster them into events — all compared against a transformer-based contextual baseline.

---

## Pipelines

### Pipeline A — Classical NLP
- **Preprocessing**: SpaCy (lemmatization, stopword removal)
- **Lexicon**: Daily Marginal Screening score `f(j) = (1/N) Σ X_k(j) · δ_k` on a 7-day rolling window
- **Embedding**: Word2Vec (Google News 300D) with lexicon filtering
- **Clustering**: Agglomerative HAC (cosine distance, average linkage)

### Pipeline B — Contextual NLP
- **Preprocessing**: BPE tokenization (Voyage-4-nano tokenizer)
- **Embedding**: DistilRoBERTa (`all-distilroberta-v1`) — full document encoding, 768D
- **Clustering**: Same HAC setup as Pipeline A

Both pipelines are evaluated on the same periods and compared on silhouette score and cluster interpretability.

---

## Data Sources

| Source | Type | Volume |
|---|---|---|
| GDELT Project (via BigQuery) | Financial news articles | 736 articles, 90 days |
| Telegram API (Telethon) | Social media / trading signals | 3,720 messages, 90 days | Not used
| Yahoo Finance (`yfinance`) | NASDAQ daily prices | 62 trading days |

**Period covered**: January 1 — March 31, 2026

---

## Project Structure

```
financial-event-clustering/
│
├── data/
│   ├── raw/                          # Raw scraped data
│   ├── processed/                    # Cleaned & tokenized news
│   │   ├── news_2026_spacy.csv       # SpaCy preprocessed
│   │   └── news_2026_bpe.csv         # BPE preprocessed
│   ├── processed/daily_lexicons_*/   # Daily lexicons (SpaCy & BPE)
│   └── for_models/                   # Document embeddings
│       ├── news_features_w2v.csv     # Word2Vec embeddings (300D)
│       └── news_features_bpe.csv     # DistilRoBERTa embeddings (768D)
│
├── notebooks/
│   ├── 0_extract_data.ipynb          # Data collection pipeline
│   ├── 1_lexicon_generation.ipynb    # Marginal Screening lexicon
│   ├── 2_feature_engineering.ipynb   # Document embeddings
│   ├── 3_news_clustering.ipynb       # HAC clustering & evaluation
│   └── 4_rag.ipynb                   # Local RAG system
│
├── src/
│   ├── extract_data.py               # GDELT scraper + Telegram collector
│   ├── lexicon_generation.py         # f(j) formula, SpaCy & BPE variants
│   ├── feature_engineering.py        # W2V and DistilRoBERTa pipelines
│   ├── news_clustering.py            # HAC, silhouette, centroids, t-SNE
│   └── llm_clustering.py             # Zero-shot LLM clustering via Ollama
│
└── requirements.txt
```

---

## Key Methods

### Marginal Screening Lexicon

For each day `d`, a lexicon is built on the 7 preceding days of news:

```
f(j) = (1/N) × Σ_k [ X_k(j) × δ_k ]
```

where `X_k(j) = 1` if word `j` appears in article `k`, and `δ_k` is the NASDAQ return the day after publication. Words above P80 or below P20 form the filtered lexicon — the ones most correlated with market movements.

### HAC Clustering with Stability

Agglomerative clustering with a stability loop: clusters smaller than `min_samples` are iteratively removed until all clusters are non-trivial. Optimal `k` is selected by silhouette maximization over `k ∈ [2, 10]`.

### Local RAG

```
User question
    ↓
DistilRoBERTa encodes the question → cosine similarity against 736 precomputed embeddings
    ↓
Top-k most relevant articles retrieved
    ↓
Qwen 2.5 0.5B generates an answer grounded in the retrieved context
```

Entirely local, zero cost, no API keys required.

---

## Installation

```bash
git clone https://github.com/your-username/financial-event-clustering.git
cd financial-event-clustering
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

For the RAG notebook, install [Ollama](https://ollama.com/download) and pull a model:

```bash
ollama pull llama3.2:3b
```

For Telegram data collection, create a `.env` file:

```env
API_ID=your_telegram_api_id
API_HASH=your_telegram_api_hash
PHONE=+33xxxxxxxxx
```

---

## Results snapshot

Example clustering on the period **Feb 25 – Mar 3, 2026** (Iran conflict week):

| Cluster | Representative headline |
|---|---|
| US–Iran geopolitical tensions | *"3 themes that drove Wall Street's wild week and the new U.S.–Iran conflict wildcard"* |
| Oil & energy shock | *"Iran war oil shock stokes fears of 1970s-style stagflation"* |
| Fed & rates reaction | *"10-year Treasury yield is little changed as oil tumbles"* |
| Tech earnings & NASDAQ | *"Nvidia leads tech rebound as investors weigh macro risks"* |

---

## Tech Stack

| Component | Tool |
|---|---|
| Data collection | `telethon`, `requests`, `yfinance`, Google BigQuery |
| NLP preprocessing | `spacy`, `transformers` (BPE tokenizer) |
| Embeddings | `gensim` (Word2Vec), `sentence-transformers` (DistilRoBERTa) |
| Clustering | `scikit-learn` (HAC, silhouette, t-SNE) |
| Visualization | `plotly` |
| LLM inference | `transformers` (Qwen 2.5 0.5B), Ollama (Llama 3.2 3B) |

---

## Authors

Roland Dutauziet & Maeva N'guessan— MSc Data Science / NLP Project, 2026




Concrètement, le projet se divise en 5 grandes étapes :

L'Extraction (Data Ingestion) : Collecte des données alternatives (articles d'actualité via GDELT, réseaux sociaux via Telegram) et des données de marché (historique du NASDAQ).

Le Lexique Sensible (Lexicon Generation) : Création d'un dictionnaire financier intelligent, capable d'identifier les mots spécifiques qui ont un impact mathématique sur les variations boursières.

La Vectorisation (Feature Engineering) : Transformation des articles en vecteurs mathématiques (Embeddings) en comparant les approches traditionnelles (Word2Vec) et modernes (Transformers).

La Détection d'Événements (Clustering) : Regroupement non supervisé des actualités (Hierarchical Agglomerative Clustering) pour identifier automatiquement les grandes thématiques ou crises du moment.

L'Analyste Virtuel (Local RAG) : Mise en place d'un système de questions-réponses interactif (Retrieval-Augmented Generation) permettant d'interroger la base de données d'événements financiers en langage naturel, avec justification des sources.
