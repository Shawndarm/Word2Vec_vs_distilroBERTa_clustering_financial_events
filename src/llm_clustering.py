import pandas as pd
import numpy as np
import json
import re
import requests
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL  = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"   # alternatives: "qwen2.5:3b", "phi3.5"
NEWS_PATH   = "../data/processed/news_2026_bpe.csv"

# ============================================================
# DATA LOADING
# ============================================================

def load_news_for_period(news_path: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Loads news articles for a given period.
    Returns a DataFrame with columns [date, headline, body].
    """
    news = pd.read_csv(news_path)
    news["date"] = pd.to_datetime(news["date"]).dt.date

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end   = datetime.strptime(end_date,   "%Y-%m-%d").date()

    mask = (news["date"] >= start) & (news["date"] <= end)
    return news[mask].reset_index(drop=True)

# ============================================================
# PROMPT BUILDER
# ============================================================

def build_clustering_prompt(news_df: pd.DataFrame, start_date: str, end_date: str) -> str:
    """
    Builds the prompt sent to the LLM.
    Uses only headlines to stay within context limits.
    """
    articles_block = "\n".join(
        f"[{i+1}] ({row['date']}) {row['headline']}"
        for i, row in news_df.iterrows()
    )

    prompt = f"""You are a financial news analyst. Below is a list of {len(news_df)} financial news headlines published between {start_date} and {end_date}.

Your task is to cluster these headlines into thematically coherent groups, where each group represents a distinct financial event or topic.

NEWS HEADLINES:
{articles_block}

Instructions:
1. Decide how many clusters best represent the distinct financial events/topics in this period (between 2 and 10).
2. Assign each article number to exactly one cluster.
3. Give each cluster a short descriptive label (e.g. "Federal Reserve Rate Decision", "Tech Earnings Season").
4. Write a 1-sentence summary of each cluster.

Respond ONLY with valid JSON in this exact format, no additional text:
{{
  "n_clusters": <integer>,
  "clusters": [
    {{
      "id": <integer starting at 0>,
      "label": "<short topic label>",
      "summary": "<1-sentence description of this financial event/topic>",
      "article_ids": [<list of article numbers from the list above>]
    }}
  ]
}}"""
    return prompt

# ============================================================
# LLM CALL
# ============================================================

def query_ollama(prompt: str, model: str = OLLAMA_MODEL, timeout: int = 120) -> str:
    """
    Sends a prompt to a local Ollama model and returns the raw response text.
    """
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,   # Low temperature for deterministic clustering
            "num_predict": 2048,
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Cannot reach Ollama. Make sure it is running: 'ollama serve' in a terminal."
        )

# ============================================================
# RESPONSE PARSER
# ============================================================

def parse_llm_response(raw_response: str, news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses the LLM JSON response and returns a DataFrame with cluster assignments.
    """
    clean = re.sub(r"
http://googleusercontent.com/immersive_entry_chip/0

Désormais, lorsque vous exécuterez la matrice de confusion (le Cross-Tab de Pandas) à l'étape 3 du notebook, vous verrez en un coup d'œil si le cluster 0 de Word2Vec/DistilRoBERTa correspond parfaitement au cluster "Tech Earnings" trouvé par le LLM !