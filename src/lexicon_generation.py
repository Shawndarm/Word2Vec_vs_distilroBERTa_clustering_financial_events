import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
import os
import re

######################  Preprocessing and Tokenisation SpaCy ######################
def preprocess_spacy(text, nlp):
    """
    Tokenization, Lemmatization,
    Stopwords removal and alpha-filtering
    """
    if not isinstance(text, str):
        return ""
    doc = nlp(text)
    # We keep only alphabetic tokens that are not stopwords and have more than 1 characters
    tokens = [
        t.lemma_.lower() for t in doc 
        if t.is_alpha and not t.is_stop and len(t.text) > 1
    ]
    return " ".join(tokens)


######################  Daily Lexicon generation ######################
def build_daily_lexicon(
    news_train,
    prices_map,
    current_date,
    dtm_output_dir,
    output_dir,
    filtered_output_dir,
):
    """
    Implements the Marginal Screening formula f(j), saves daily lexicon and filtered daily lexicon (P20 & P80),
    and exports the Document-Term Matrix (DTM) for visualization.
    """
    N = len(news_train)

    # a. Vectorization (Binary 0/1) and Frequency Filtering: max_df=0.9, min_df=10
    vectorizer = CountVectorizer(
        binary=True,  # Use a dummy variable X_k(j)
        max_df=0.90,
        min_df=5,
        stop_words="english",
    )
    try:
        dtm_sparse = vectorizer.fit_transform(news_train["clean"])
        words = vectorizer.get_feature_names_out()
    except ValueError:
        return None

    # b. Document-Term Matrix (DTM)
    os.makedirs(dtm_output_dir, exist_ok=True)
    # We create a DataFrame where rows are news articles and columns are words
    dtm_df = pd.DataFrame(dtm_sparse.toarray(), columns=words)
    dtm_df.insert(0, "article_date", news_train["date"].values)
    # We add the dates to make the visualization more informative
    dtm_path = os.path.join(dtm_output_dir, f"dtm_{current_date}.csv")
    dtm_df.to_csv(dtm_path, index=False)

    # c. Marginal Screening Formula f(j) : f(j) = (1/N) * sum( X_k(j) * delta_k )
    deltas = news_train["date"].map(prices_map).fillna(0).values
    sum_product = np.array(dtm_sparse.T.dot(deltas)).flatten()
    f_j = sum_product / N

    # d. Saving complete lexicon (for Plotly visualizations)
    lexicon_df = pd.DataFrame({"word": words, "score": f_j})
    os.makedirs(output_dir, exist_ok=True)
    full_lex_path = os.path.join(output_dir, f"lexicon_full_{current_date}.csv")
    lexicon_df.sort_values("score", ascending=False).to_csv(full_lex_path, index=False)

    # e. Percentiles (P20 & P80) based selection and saving filtered lexicon
    p20 = np.percentile(f_j, 20)
    p80 = np.percentile(f_j, 80)

    filtered_lex_df = lexicon_df[
        (lexicon_df["score"] >= p80) | (lexicon_df["score"] <= p20)
    ]
    os.makedirs(filtered_output_dir, exist_ok=True)
    filtered_path = os.path.join(
        filtered_output_dir, f"lexicon_filtered_{current_date}.csv"
    )
    filtered_lex_df.sort_values("score", ascending=False).to_csv(
        filtered_path, index=False
    )
    return


######################  Lexicon visualization ######################
def visualize_daily_lexicon(date_str):
    """
    Loads a daily lexicon CSV and plots the f(j) distribution with thresholds.
    """
    FILE_PATH = f"../data/processed/daily_lexicons_full_spacy/lexicon_full_{date_str}.csv"

    # Load and sort data
    df = pd.read_csv(FILE_PATH)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    # Re-calculate thresholds for the visualization
    p20 = np.percentile(df["score"], 20)
    p80 = np.percentile(df["score"], 80)
    # Categorize words for coloring
    df["type"] = "Neutral"
    df.loc[df["score"] >= p80, "type"] = "Positive"
    df.loc[df["score"] <= p20, "type"] = "Negative"

    # Bar Chart
    fig = px.bar(
        df,
        x="word",
        y="score",
        color="type",
        color_discrete_map={
            "Positive": "#2ecc71",
            "Negative": "#e74c3c",
            "Neutral": "#bdc3c7",
        },
        title=f"Lexicon Sentiment Scores (f_j) - {date_str}",
        labels={"score": "Score f(j)", "word": "Financial Terms", "type": "Category"},
        hover_data={"score": ":.5f"},
    )
    # Add Horizontal Lines for P80 and P20
    fig.add_hline(
        y=p80,
        line_dash="dash",
        line_color="#27ae60",
        annotation_text=f"P80 Threshold ({p80:.5f})",
        annotation_position="top right",
    )
    fig.add_hline(
        y=p20,
        line_dash="dash",
        line_color="#c0392b",
        annotation_text=f"P20 Threshold ({p20:.5f})",
        annotation_position="bottom right",
    )
    # Styling: Add Range Slider because there are many words
    fig.update_layout(
        xaxis_title="Words (Sorted by Score)",
        yaxis_title="Marginal Screening Score f(j)",
        xaxis_tickangle=-45,
        height=600,
        template="plotly_white",
    )
    fig.show()


####################### BPE versions of the above functions #######################


def preprocess_bpe(text, tokenizer):
    """
    Cleans the text (removes numbers and punctuation) and then applies BPE.
    """
    if not isinstance(text, str):
        return ""   
    # 1. CLEANING (Regular Expression)
    # [^a-zA-ZÀ-ÿ\s] : we keep only letters (including accented ones) and spaces.
    # Everything else (numbers, parentheses, dashes, %, etc.) is replaced with a space.
    clean_text = re.sub(r'[^a-zA-ZÀ-ÿ\s]', ' ', text)
    # Delete spaces at the beginning and end, and reduce multiple spaces to a single space
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()   
    # 2. TOKENISATION BPE
    tokens = tokenizer.tokenize(clean_text)
    return " ".join(tokens)

######################  Daily Lexicon generation (BPE Aware) ######################
def build_daily_lexicon_bpe(
    news_train,
    prices_map,
    current_date,
    dtm_output_dir,
    output_dir,
    filtered_output_dir,
):
    """
    Implémente la formule de Marginal Screening f(j), sauvegarde le lexique quotidien 
    et exporte la DTM. Adapté pour gérer les jetons BPE.
    """
    N = len(news_train)

    # a. Vectorization (Binary 0/1) and Frequency Filtering
    # Changements critiques ici pour le BPE :
    vectorizer = CountVectorizer(
        binary=True,  
        max_df=0.90,
        min_df=5,
        lowercase=False, # IMPORTANT : Ne pas modifier la casse des jetons BPE
        token_pattern=None, # IMPORTANT : Désactiver le regex par défaut
        tokenizer=lambda x: x.split(), # IMPORTANT : Séparer uniquement par les espaces qu'on a ajoutés
        # On retire stop_words="english" car les stopwords BPE ont souvent un "Ġ" devant
        # (ex: "Ġthe"), ce qui fait que la liste standard de sklearn ne les trouvera pas.
    )
    
    try:
        dtm_sparse = vectorizer.fit_transform(news_train["clean"])
        words = vectorizer.get_feature_names_out()
    except ValueError:
        return None

    # b. Document-Term Matrix (DTM)
    os.makedirs(dtm_output_dir, exist_ok=True)
    dtm_df = pd.DataFrame(dtm_sparse.toarray(), columns=words)
    dtm_df.insert(0, "article_date", news_train["date"].values)
    
    dtm_path = os.path.join(dtm_output_dir, f"dtm_bpe_{current_date}.csv")
    dtm_df.to_csv(dtm_path, index=False)

    # c. Marginal Screening Formula f(j): 
    # Formule : f(j) = (1/N) * sum( X_k(j) * delta_k )
    deltas = news_train["date"].map(prices_map).fillna(0).values
    sum_product = np.array(dtm_sparse.T.dot(deltas)).flatten()
    f_j = sum_product / N

    # d. Saving complete lexicon
    lexicon_df = pd.DataFrame({"word": words, "score": f_j})
    os.makedirs(output_dir, exist_ok=True)
    full_lex_path = os.path.join(output_dir, f"lexicon_bpe_full_{current_date}.csv")
    lexicon_df.sort_values("score", ascending=False).to_csv(full_lex_path, index=False)

    # e. Percentiles (P20 & P80)
    p20 = np.percentile(f_j, 20)
    p80 = np.percentile(f_j, 80)

    filtered_lex_df = lexicon_df[
        (lexicon_df["score"] >= p80) | (lexicon_df["score"] <= p20)
    ]
    os.makedirs(filtered_output_dir, exist_ok=True)
    filtered_path = os.path.join(
        filtered_output_dir, f"lexicon_bpe_filtered_{current_date}.csv"
    )
    filtered_lex_df.sort_values("score", ascending=False).to_csv(
        filtered_path, index=False
    )
    return


######################  Lexicon visualization (BPE Aware) ######################
def visualize_daily_lexicon_bpe(date_str):
    """
    Charge un CSV de lexique BPE quotidien et trace la distribution de f(j).
    """

    FILE_PATH = f"../data/processed/daily_lexicons_full_bpe/lexicon_bpe_full_{date_str}.csv"

    df = pd.read_csv(FILE_PATH)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    
    p20 = np.percentile(df["score"], 20)
    p80 = np.percentile(df["score"], 80)
    
    df["type"] = "Neutral"
    df.loc[df["score"] >= p80, "type"] = "Positive"
    df.loc[df["score"] <= p20, "type"] = "Negative"

    fig = px.bar(
        df,
        x="word",
        y="score",
        color="type",
        color_discrete_map={
            "Positive": "#2ecc71",
            "Negative": "#e74c3c",
            "Neutral": "#bdc3c7",
        },
        title=f"BPE Sub-word Sentiment Scores (f_j) - {date_str}",
        labels={"score": "Score f(j)", "word": "BPE Tokens", "type": "Category"},
        hover_data={"score": ":.5f"},
    )
    
    fig.add_hline(
        y=p80, line_dash="dash", line_color="#27ae60",
        annotation_text=f"P80 Threshold ({p80:.5f})", annotation_position="top right",
    )
    fig.add_hline(
        y=p20, line_dash="dash", line_color="#c0392b",
        annotation_text=f"P20 Threshold ({p20:.5f})", annotation_position="bottom right",
    )
    
    fig.update_layout(
        xaxis_title="BPE Tokens (Sorted by Score)",
        yaxis_title="Marginal Screening Score f(j)",
        xaxis_tickangle=-45,
        height=600,
        template="plotly_white",
    )
    fig.show()