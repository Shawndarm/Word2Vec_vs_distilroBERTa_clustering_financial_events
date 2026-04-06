"""
Pipeline comparatif : SpaCy+Word2Vec  vs  BPE+FinBERT
=======================================================
Objectif : comparer deux approches d'embedding pour le clustering
           d'événements financiers NASDAQ Jan-Mars 2026.

Ce qu'on compare :
  - Corrélation signal de sentiment vs returns NASDAQ
  - Qualité des clusters (silhouette, Davies-Bouldin)
  - Capacité à détecter les événements financiers connus

pip install spacy transformers torch pandas numpy scikit-learn
    sentence-transformers plotly tqdm
python -m spacy download en_core_web_sm
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

def load_data(tweets_path, news_path, nasdaq_path):
    tweets = pd.read_csv(tweets_path)
    news   = pd.read_csv(news_path)
    nasdaq = pd.read_csv(nasdaq_path)

    # Nettoyage nasdaq (supprimer les 2 premières lignes parasites)
    nasdaq = nasdaq[~nasdaq["Price"].isin(["Ticker", "Date"])].copy()
    nasdaq = nasdaq[nasdaq["Close"].notna()].copy()
    nasdaq = nasdaq.rename(columns={"Price": "date"})
    nasdaq["date"]    = pd.to_datetime(nasdaq["date"])
    nasdaq["Close"]   = pd.to_numeric(nasdaq["Close"], errors="coerce")
    nasdaq["returns"] = nasdaq["returns"].astype(float)
    nasdaq = nasdaq.dropna(subset=["returns"])

    # Fusionner tweets + news en un seul corpus
    tweets["source_type"] = "social"
    news["source_type"]   = "news"
    news["text"] = news["headline"] + " " + news["body"]

    # Colonnes communes
    corpus = pd.concat([
        tweets[["date", "text", "source_type"]],
        news[["date",   "text", "source_type"]],
    ], ignore_index=True)
    corpus["date"] = pd.to_datetime(corpus["date"])
    corpus = corpus[corpus["text"].notna() & (corpus["text"].str.strip() != "")]

    return corpus, nasdaq


# ============================================================
# PIPELINE A : SpaCy + Lexique (formule f(j)) + Word2Vec
# ============================================================

class PipelineSpacyW2V:
    """
    Pipeline classique :
    1. Preprocessing SpaCy (lemmatisation, stopwords, alpha-filter)
    2. Lexique financier via formule f(j) de Marginal Screening
    3. Embeddings Word2Vec entraîné sur le corpus
    4. Score journalier = moyenne des vecteurs pondérés par f(j)
    """

    def __init__(self):
        import spacy
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.w2v_model   = None
        self.lexicon     = None

    # ----------------------------------------------------------
    # 1. Preprocessing SpaCy
    # ----------------------------------------------------------

    def preprocess(self, text: str) -> str:
        """Lemmatisation + suppression stopwords + filtre alpha."""
        if not isinstance(text, str):
            return ""
        doc = self.nlp(text.lower())
        tokens = [
            t.lemma_ for t in doc
            if t.is_alpha and not t.is_stop and len(t.text) > 2
        ]
        return " ".join(tokens)

    def preprocess_corpus(self, corpus: pd.DataFrame) -> pd.DataFrame:
        print("🔤 [A] Preprocessing SpaCy...")
        corpus = corpus.copy()
        corpus["clean"] = [
            self.preprocess(t) for t in tqdm(corpus["text"], desc="SpaCy")
        ]
        return corpus

    # ----------------------------------------------------------
    # 2. Lexique f(j) — Marginal Screening
    # ----------------------------------------------------------

    def build_lexicon(self, corpus: pd.DataFrame, prices_map: dict) -> pd.DataFrame:
        """
        f(j) = (1/N) * Σ_k [ X_k(j) * δ_k ]
        X_k(j) = 1 si le mot j apparaît dans le doc k, 0 sinon
        δ_k    = return du NASDAQ le jour du doc k
        """
        from sklearn.feature_extraction.text import CountVectorizer

        print("📚 [A] Construction du lexique f(j)...")
        corpus = corpus[corpus["clean"].str.strip() != ""].copy()
        N = len(corpus)

        vectorizer = CountVectorizer(
            binary=True,
            max_df=0.90,
            min_df=5,              # réduit à 5 pour plus de couverture
            stop_words="english",
        )
        dtm = vectorizer.fit_transform(corpus["clean"])
        words = vectorizer.get_feature_names_out()

        deltas = corpus["date"].map(prices_map).fillna(0).values
        f_j    = dtm.T.dot(deltas).ravel() / N

        self.lexicon = pd.DataFrame({"word": words, "score": f_j})
        print(f"   Lexique : {len(self.lexicon)} mots | "
              f"pos={( self.lexicon['score'] > 0).sum()} | "
              f"neg={(self.lexicon['score'] < 0).sum()}")
        return self.lexicon

    # ----------------------------------------------------------
    # 3. Word2Vec sur le corpus
    # ----------------------------------------------------------

    def train_word2vec(self, corpus: pd.DataFrame):
        from gensim.models import Word2Vec

        print("🧠 [A] Entraînement Word2Vec...")
        sentences = [
            text.split() for text in corpus["clean"]
            if isinstance(text, str) and text.strip()
        ]
        self.w2v_model = Word2Vec(
            sentences,
            vector_size=100,
            window=5,
            min_count=3,
            workers=4,
            epochs=10,
        )
        print(f"   Vocabulaire W2V : {len(self.w2v_model.wv)} mots")

    # ----------------------------------------------------------
    # 4. Score journalier
    # ----------------------------------------------------------

    def doc_embedding(self, text: str) -> np.ndarray:
        """
        Embedding d'un doc = moyenne des vecteurs W2V des mots,
        pondérée par |f(j)| (les mots du lexique comptent plus).
        """
        if self.w2v_model is None or self.lexicon is None:
            return np.zeros(100)

        lex_dict = dict(zip(self.lexicon["word"], self.lexicon["score"]))
        words    = text.split() if isinstance(text, str) else []
        vectors, weights = [], []

        for word in words:
            if word in self.w2v_model.wv:
                w = abs(lex_dict.get(word, 0.01))  # poids minimal pour les mots hors lexique
                vectors.append(self.w2v_model.wv[word] * w)
                weights.append(w)

        if not vectors:
            return np.zeros(100)

        return np.sum(vectors, axis=0) / (sum(weights) + 1e-9)

    def sentiment_score(self, text: str) -> float:
        """Score scalaire = projection sur l'axe positif/négatif du lexique."""
        if not isinstance(text, str):
            return 0.0
        lex_dict = dict(zip(self.lexicon["word"], self.lexicon["score"]))
        words    = text.split()
        scores   = [lex_dict[w] for w in words if w in lex_dict]
        return float(np.mean(scores)) if scores else 0.0

    def compute_daily_signals(self, corpus: pd.DataFrame) -> pd.DataFrame:
        """Agrège les scores par jour."""
        print("📈 [A] Calcul des signaux journaliers...")
        corpus = corpus[corpus["clean"].str.strip() != ""].copy()
        corpus["sentiment_A"] = corpus["clean"].apply(self.sentiment_score)
        daily = corpus.groupby("date")["sentiment_A"].mean().reset_index()
        daily.columns = ["date", "sentiment_A"]
        return daily

    def compute_daily_embeddings(self, corpus: pd.DataFrame) -> pd.DataFrame:
        """Embedding moyen par jour (pour le clustering)."""
        print("🔢 [A] Calcul des embeddings journaliers...")
        corpus = corpus[corpus["clean"].str.strip() != ""].copy()
        embeds = np.array([self.doc_embedding(t) for t in tqdm(corpus["clean"], desc="W2V embed")])
        corpus = corpus.copy()
        corpus["embed_A"] = list(embeds)
        daily_embed = corpus.groupby("date")["embed_A"].apply(
            lambda x: np.mean(np.stack(x.values), axis=0)
        ).reset_index()
        return daily_embed


# ============================================================
# PIPELINE B : BPE + FinBERT
# ============================================================

class PipelineFinBERT:
    """
    Pipeline moderne :
    1. Tokenisation BPE interne à FinBERT (automatique via HuggingFace)
    2. Embeddings [CLS] token de FinBERT (768 dimensions)
    3. Score de sentiment via la tête de classification FinBERT
       (positive / negative / neutral)
    4. Score journalier = moyenne pondérée des scores docs
    """

    # FinBERT = BERT fine-tuné sur textes financiers (ProsusAI)
    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self, device=None):
        import torch
        from transformers import AutoTokenizer, AutoModel, pipeline

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🤖 [B] Chargement FinBERT sur {self.device}...")

        # Tokenizer BPE (WordPiece en réalité pour BERT, sous-mots)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        # Modèle pour les embeddings
        self.model     = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device)
        self.model.eval()
        # Pipeline sentiment (positive/negative/neutral)
        self.sentiment_pipeline = pipeline(
            "text-classification",
            model=self.MODEL_NAME,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            truncation=True,
            max_length=512,
        )
        print("   FinBERT chargé ✅")

    # ----------------------------------------------------------
    # 1. Embedding [CLS] d'un texte
    # ----------------------------------------------------------

    def get_embedding(self, text: str) -> np.ndarray:
        """Retourne le vecteur [CLS] de FinBERT (768d)."""
        import torch

        if not isinstance(text, str) or not text.strip():
            return np.zeros(768)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # [CLS] = premier token, représente le document entier
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return cls_embedding

    # ----------------------------------------------------------
    # 2. Score de sentiment FinBERT
    # ----------------------------------------------------------

    def get_sentiment(self, text: str) -> float:
        """
        Retourne un score scalaire :
        +1 = positive, -1 = negative, 0 = neutral
        Pondéré par la probabilité du modèle.
        """
        if not isinstance(text, str) or not text.strip():
            return 0.0
        try:
            result = self.sentiment_pipeline(text[:512])[0]
            label, score = result["label"], result["score"]
            if label == "positive":
                return score
            elif label == "negative":
                return -score
            else:
                return 0.0
        except:
            return 0.0

    # ----------------------------------------------------------
    # 3. Signaux journaliers
    # ----------------------------------------------------------

    def compute_daily_signals(self, corpus: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
        """Score de sentiment moyen par jour."""
        print("📈 [B] Calcul des sentiments FinBERT...")

        texts  = corpus["text"].fillna("").tolist()
        scores = []

        for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT sentiment"):
            batch = [t[:512] for t in texts[i:i+batch_size]]
            try:
                results = self.sentiment_pipeline(batch)
                for r in results:
                    label, prob = r["label"], r["score"]
                    if label == "positive":   scores.append(prob)
                    elif label == "negative": scores.append(-prob)
                    else:                     scores.append(0.0)
            except:
                scores.extend([0.0] * len(batch))

        corpus = corpus.copy()
        corpus["sentiment_B"] = scores
        daily  = corpus.groupby("date")["sentiment_B"].mean().reset_index()
        daily.columns = ["date", "sentiment_B"]
        return daily

    def compute_daily_embeddings(self, corpus: pd.DataFrame) -> pd.DataFrame:
        """Embedding [CLS] moyen par jour."""
        print("🔢 [B] Calcul des embeddings FinBERT...")
        embeds = np.array([
            self.get_embedding(t)
            for t in tqdm(corpus["text"].fillna(""), desc="FinBERT embed")
        ])
        corpus = corpus.copy()
        corpus["embed_B"] = list(embeds)
        daily_embed = corpus.groupby("date")["embed_B"].apply(
            lambda x: np.mean(np.stack(x.values), axis=0)
        ).reset_index()
        return daily_embed


# ============================================================
# COMPARAISON ET ÉVALUATION
# ============================================================

class PipelineEvaluator:
    """Compare les deux pipelines sur 3 critères :
    1. Corrélation signal de sentiment vs returns
    2. Qualité du clustering (silhouette)
    3. Détection des événements connus
    """

    # Événements NASDAQ Jan-Mars 2026 à détecter
    KNOWN_EVENTS = [
        {"date": "2026-01-20", "label": "Inauguration Trump",      "type": "politique"},
        {"date": "2026-01-29", "label": "Fed réunion FOMC",         "type": "macro"},
        {"date": "2026-02-19", "label": "Correction NASDAQ -3%",    "type": "marché"},
        {"date": "2026-03-04", "label": "Annonce tarifs douaniers", "type": "macro"},
        {"date": "2026-03-18", "label": "FOMC + dot plot",          "type": "macro"},
    ]

    def __init__(self, nasdaq: pd.DataFrame):
        self.nasdaq = nasdaq
        self.nasdaq["date"] = pd.to_datetime(self.nasdaq["date"])

    def evaluate_correlation(self, daily_A: pd.DataFrame, daily_B: pd.DataFrame) -> dict:
        """
        Corrélation de Pearson et Spearman entre sentiment et returns.
        On teste aussi le lag +1 (le sentiment d'hier prédit-il le return d'aujourd'hui ?)
        """
        from scipy.stats import pearsonr, spearmanr

        merged_A = daily_A.merge(self.nasdaq[["date", "returns"]], on="date", how="inner")
        merged_B = daily_B.merge(self.nasdaq[["date", "returns"]], on="date", how="inner")

        results = {}
        for name, merged, col in [
            ("SpaCy+W2V", merged_A, "sentiment_A"),
            ("FinBERT",   merged_B, "sentiment_B"),
        ]:
            s = merged[col].values
            r = merged["returns"].values

            # Lag 0 : même jour
            p0, _ = pearsonr(s, r)
            sp0, _ = spearmanr(s, r)

            # Lag +1 : sentiment j-1 prédit return j
            if len(s) > 1:
                p1, _ = pearsonr(s[:-1], r[1:])
                sp1, _ = spearmanr(s[:-1], r[1:])
            else:
                p1 = sp1 = 0.0

            results[name] = {
                "pearson_lag0":  round(p0,  4),
                "spearman_lag0": round(sp0, 4),
                "pearson_lag1":  round(p1,  4),
                "spearman_lag1": round(sp1, 4),
            }
            print(f"\n📊 {name}")
            print(f"   Pearson  lag0={p0:.4f}  lag1={p1:.4f}")
            print(f"   Spearman lag0={sp0:.4f}  lag1={sp1:.4f}")

        return results

    def evaluate_clustering(
        self,
        embed_A: pd.DataFrame,
        embed_B: pd.DataFrame,
        n_clusters: int = 5,
    ) -> dict:
        """
        K-Means sur les embeddings journaliers.
        Métriques : silhouette score, Davies-Bouldin index.
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        from sklearn.preprocessing import StandardScaler

        results = {}
        for name, embed_df, col in [
            ("SpaCy+W2V", embed_A, "embed_A"),
            ("FinBERT",   embed_B, "embed_B"),
        ]:
            X = np.stack(embed_df[col].values)
            X_scaled = StandardScaler().fit_transform(X)

            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)

            sil = silhouette_score(X_scaled, labels)
            db  = davies_bouldin_score(X_scaled, labels)

            results[name] = {
                "silhouette":      round(sil, 4),
                "davies_bouldin":  round(db,  4),
                "n_clusters":      n_clusters,
            }
            print(f"\n🗂️  {name} — {n_clusters} clusters")
            print(f"   Silhouette      : {sil:.4f}  (plus proche de 1 = mieux)")
            print(f"   Davies-Bouldin  : {db:.4f}   (plus proche de 0 = mieux)")

            # Ajouter les labels au dataframe
            embed_df[f"cluster_{name.replace('+','_')}"] = labels

        return results

    def plot_comparison(
        self,
        daily_A: pd.DataFrame,
        daily_B: pd.DataFrame,
    ):
        """Visualisation des signaux de sentiment vs returns NASDAQ."""

        merged = (
            daily_A
            .merge(daily_B, on="date", how="outer")
            .merge(self.nasdaq[["date", "returns"]], on="date", how="left")
            .sort_values("date")
        )

        fig = go.Figure()

        # Returns NASDAQ
        fig.add_trace(go.Scatter(
            x=merged["date"], y=merged["returns"],
            name="Returns NASDAQ", line=dict(color="#2c3e50", width=1.5),
            yaxis="y2"
        ))

        # Sentiment SpaCy+W2V
        if "sentiment_A" in merged.columns:
            fig.add_trace(go.Scatter(
                x=merged["date"], y=merged["sentiment_A"],
                name="Sentiment SpaCy+W2V",
                line=dict(color="#3498db", width=1.5, dash="dot"),
            ))

        # Sentiment FinBERT
        if "sentiment_B" in merged.columns:
            fig.add_trace(go.Scatter(
                x=merged["date"], y=merged["sentiment_B"],
                name="Sentiment FinBERT",
                line=dict(color="#e74c3c", width=1.5),
            ))

        # Événements connus
        for evt in self.KNOWN_EVENTS:
            x_date = pd.to_datetime(evt["date"])

            # Ligne verticale
            fig.add_vline(
                x=x_date,
                line_dash="dash",
                line_color="#f39c12",
            )

            # Annotation séparée (SAFE)
            fig.add_annotation(
                x=x_date,
                y=1,  # en haut du plot
                yref="paper",
                text=evt["label"],
                showarrow=False,
                yshift=10,
                font=dict(size=10, color="#f39c12"),
            )

        fig.update_layout(
            title="Signaux de sentiment vs Returns NASDAQ (Jan-Mars 2026)",
            yaxis=dict(title="Sentiment score"),
            yaxis2=dict(title="Returns", overlaying="y", side="right"),
            template="plotly_white",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig.show()
        return fig


# ============================================================
# RUNNER PRINCIPAL
# ============================================================

def run_comparison(
    tweets_path="data/raw/raw_tweets_2026.csv",
    news_path="data/raw/raw_news_2026.csv",
    nasdaq_path="data/raw/raw_nasdaq_2026.csv",
    run_finbert=True,   # mettre False si pas de GPU / pas de temps
    n_clusters=5,
):
    print("=" * 60)
    print("  COMPARAISON : SpaCy+W2V  vs  BPE+FinBERT")
    print("=" * 60)

    # 1. Chargement
    corpus, nasdaq = load_data(tweets_path, news_path, nasdaq_path)
    prices_map = dict(zip(
        pd.to_datetime(nasdaq["date"]),
        nasdaq["returns"]
    ))

    evaluator = PipelineEvaluator(nasdaq)

    # ──────────────────────────────────────────────────────────
    # PIPELINE A : SpaCy + Word2Vec
    # ──────────────────────────────────────────────────────────
    print("\n" + "─"*40)
    print("  PIPELINE A : SpaCy + Word2Vec")
    print("─"*40)

    pip_a = PipelineSpacyW2V()
    corpus_a = pip_a.preprocess_corpus(corpus)
    pip_a.build_lexicon(corpus_a, prices_map)
    pip_a.train_word2vec(corpus_a)

    daily_sentiment_A  = pip_a.compute_daily_signals(corpus_a)
    daily_embedding_A  = pip_a.compute_daily_embeddings(corpus_a)

    # ──────────────────────────────────────────────────────────
    # PIPELINE B : FinBERT
    # ──────────────────────────────────────────────────────────
    daily_sentiment_B = pd.DataFrame(columns=["date", "sentiment_B"])
    daily_embedding_B = pd.DataFrame(columns=["date", "embed_B"])

    if run_finbert:
        print("\n" + "─"*40)
        print("  PIPELINE B : FinBERT")
        print("─"*40)
        pip_b = PipelineFinBERT()
        # Pour les embeddings, on sous-échantillonne si nécessaire (FinBERT est lent sur CPU)
        corpus_sample = pd.concat([
            group.sample(min(len(group), 20), random_state=42)
            for _, group in corpus.groupby("date")
        ]).reset_index(drop=True)
        print(corpus_sample.columns)

        daily_sentiment_B = pip_b.compute_daily_signals(corpus_sample)
        daily_embedding_B = pip_b.compute_daily_embeddings(corpus_sample)

    # ──────────────────────────────────────────────────────────
    # ÉVALUATION
    # ──────────────────────────────────────────────────────────
    print("\n" + "="*40)
    print("  ÉVALUATION")
    print("="*40)

    print("\n[1] Corrélation sentiment vs returns NASDAQ")
    corr = evaluator.evaluate_correlation(daily_sentiment_A, daily_sentiment_B)

    if run_finbert and len(daily_embedding_B) > 0:
        print("\n[2] Qualité du clustering")
        clust = evaluator.evaluate_clustering(
            daily_embedding_A, daily_embedding_B, n_clusters=n_clusters
        )

    print("\n[3] Visualisation")
    evaluator.plot_comparison(daily_sentiment_A, daily_sentiment_B)

    return {
        "sentiment_A":  daily_sentiment_A,
        "sentiment_B":  daily_sentiment_B,
        "embedding_A":  daily_embedding_A,
        "embedding_B":  daily_embedding_B,
        "correlations": corr,
    }


if __name__ == "__main__":
    results = run_comparison(
        tweets_path="data/raw/raw_tweets_2026.csv",
        news_path="data/raw/raw_news_2026.csv",
        nasdaq_path="data/raw/raw_nasdaq_2026.csv",
        run_finbert=True,
        n_clusters=3,
    )

