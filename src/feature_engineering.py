import pandas as pd
import numpy as np
from tqdm import tqdm
import os


######### News Embedding Function #########
def compute_news_embedding(text, lexicon_set, model):
    """
    Core Feature Engineering Step:
    1. Filter: Retain only words present in the daily lexicon.
    2. Embedding: Extract vectors for these specific words.
    3. Average: Compute the mean vector (News-Embedding).
    """
    tokens = text.split()
    # Authors' Logic: "only the words that appear in the lexicon are retained"
    # We also check if the word exists in the Dolma model vocabulary
    valid_vectors = [model[w] for w in tokens if w in lexicon_set and w in model]
    # If no words match the lexicon for this article, return a neutral zero vector
    if not valid_vectors:
        return np.zeros(300)
    # Average the vectors to get the document representation in 300D space
    return np.mean(valid_vectors, axis=0)


############# Main Pipeline Function #########
def run_feature_engineering_pipeline(news_df, lexicon_dir, model):
    """
    Iterates through the corpus and applies the daily lexicon filtering
    to generate semantic embeddings for each document.
    """
    embedded_data = []
    # Process day by day to align news with their specific daily lexicon
    unique_days = sorted(news_df["date"].unique())

    for current_day in tqdm(unique_days, desc="Feature Engineering"):
        lex_path = os.path.join(lexicon_dir, f"lexicon_filtered_{current_day}.csv")

        if os.path.exists(lex_path):
            # Load the lexicon produced for THIS specific day
            day_lexicon = set(pd.read_csv(lex_path)["word"].tolist())
            # Filter news published on THIS specific day
            daily_news = news_df[news_df["date"] == current_day]

            for idx, row in daily_news.iterrows():
                # Generate the 300D News-Embedding
                vector = compute_news_embedding(row["clean"], day_lexicon, model)
                embedded_data.append(
                    {
                        "date": current_day,
                        "embedding": vector,
                        "headline": row[
                            "headline"
                        ],  # we keep headline for later cluster audit
                    }
                )

    return pd.DataFrame(embedded_data)


############# Main Pipeline Function for DistilBERT #########
def run_document_embedding_bpe(news_df, model, batch_size=16):
    """
    Embeds ALL documents in the dataset without any lexicon filtering.
    Optimized for speed using batch processing on CPU.
    """
    embedded_data = []

    # Cleaning: Remove empty texts
    valid_news = news_df.dropna(subset=["clean"]).copy()
    valid_news = valid_news[valid_news["clean"].str.strip() != ""]

    print(f"Starting batch embedding for {len(valid_news)} articles...")

    # Extract texts for encoding
    texts_to_encode = valid_news["clean"].tolist()

    # BATCH EMBEDDING
    embeddings = model.encode(
        texts_to_encode, batch_size=batch_size, show_progress_bar=True
    )

    # Reconstruction: Mapping vectors back to dates and headlines
    for idx, (row_idx, row) in enumerate(valid_news.iterrows()):
        embedded_data.append(
            {
                "date": row["date"],
                "embedding": embeddings[idx],
                "headline": row["headline"],
            }
        )

    return pd.DataFrame(embedded_data)
