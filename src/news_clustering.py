import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import silhouette_samples
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


# --- 1. FONCTION DE PARSING DES EMBEDDINGS ---
def parse_embedding_string(emb_str):
    """
    Convertit la chaîne de caractères du CSV (ex: "[ 0.12 \n -0.34 ...]")
    en un vrai tableau numpy utilisable pour le clustering.
    """
    clean_str = str(emb_str).replace("[", "").replace("]", "").replace("\n", " ")
    return np.fromstring(clean_str, sep=" ")


################# Clustering evaluation (HAC Only - Period Based) ##############
def run_hac_evaluation_period(
    X_full, df_features, start_date, end_date, k_range=range(2, 11), min_samples=2
):
    """
    Évalue le HAC sur une période temporelle stricte pour trouver le K optimal.
    """
    # 1. Filtrage temporel
    mask = (df_features["date"] >= start_date) & (df_features["date"] <= end_date)
    X_sub = X_full[mask.values]

    if len(X_sub) < max(k_range):
        print(
            f"Attention: Pas assez de documents ({len(X_sub)}) pour tester k={max(k_range)}."
        )
        k_range = range(2, max(2, len(X_sub)))  # Sécurité

    results = {"k": [], "Silhouette_Score": []}

    for k in k_range:
        results["k"].append(k)

        try:
            curr_X = X_sub.copy()
            labels_hac = None

            # Logique de stabilité
            while True:
                if len(curr_X) < k:
                    labels_hac = None
                    break

                model = AgglomerativeClustering(
                    n_clusters=k, metric="cosine", linkage="average"
                )
                temp_labels = model.fit_predict(curr_X)
                counts = pd.Series(temp_labels).value_counts()
                to_remove = counts[counts < min_samples].index

                if len(to_remove) == 0:
                    labels_hac = temp_labels
                    break

                curr_X = curr_X[~np.isin(temp_labels, to_remove)]

            # Score de silhouette
            if labels_hac is not None and len(np.unique(labels_hac)) > 1:
                score = silhouette_score(curr_X, labels_hac, metric="cosine")
                results["Silhouette_Score"].append(score)
            else:
                results["Silhouette_Score"].append(np.nan)

        except Exception:
            results["Silhouette_Score"].append(np.nan)

    return pd.DataFrame(results)


def plot_hac_evaluation(results_df, title="HAC Silhouette Scores"):
    """Affiche le graphique d'évaluation"""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=results_df["k"],
            y=results_df["Silhouette_Score"],
            mode="lines+markers",
            name="HAC Score",
            line=dict(color="#2ecc71", width=2),
            marker=dict(symbol="circle", size=10),
        )
    )
    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Average Silhouette Score",
        template="plotly_white",
    )
    fig.show()
    return fig


################## Visualization of HAC clusters in 2D with Plotly ##############
def visualize_hac_tsne_range(
    X_full, df_features, start_date, end_date, k, perplexity=30, min_samples=2
):
    """
    Filters data, applies stability logic, and shows t-SNE visualization.
    """
    mask = (df_features["date"] >= start_date) & (df_features["date"] <= end_date)
    sub_df = df_features.loc[mask].copy()
    X_sub = X_full[mask.values]

    # HAC Stability Logic
    curr_X = X_sub.copy()
    curr_indices = np.arange(len(sub_df))
    final_labels = None

    while True:
        if len(curr_X) < k:
            print(f"Could not achieve a stable configuration for k={k}.")
            return None
        model = AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average"
        )
        temp_labels = model.fit_predict(curr_X)
        counts = pd.Series(temp_labels).value_counts()
        to_remove = counts[counts < min_samples].index

        if len(to_remove) == 0:
            final_labels = temp_labels
            break

        mask_stable = ~np.isin(temp_labels, to_remove)
        curr_X = curr_X[mask_stable]
        curr_indices = curr_indices[mask_stable]

    stable_df = sub_df.iloc[curr_indices].copy()
    stable_df["Cluster"] = final_labels.astype(str)

    actual_perplexity = min(perplexity, max(1, len(curr_X) - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=actual_perplexity,
        random_state=42,
        init="pca",
        metric="cosine",
    )
    X_2d = tsne.fit_transform(curr_X)

    stable_df["X"] = X_2d[:, 0]
    stable_df["Y"] = X_2d[:, 1]

    fig = px.scatter(
        stable_df,
        x="X",
        y="Y",
        color="Cluster",
        hover_data={"headline": True, "date": True, "X": False, "Y": False},
        title=f"Stable Financial Events ({start_date} to {end_date}) | k={k}",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )
    fig.update_traces(
        marker=dict(size=12, opacity=0.8, line=dict(width=1, color="white"))
    )
    return fig


################## Final HAC & Dendrogram ##############
def compute_stable_hac_linkage(
    X_full, df_features, start_date, end_date, k, min_samples=2
):
    mask = (df_features["date"] >= start_date) & (df_features["date"] <= end_date)
    X_sub = X_full[mask.values]
    sub_df = df_features.loc[mask].copy()

    curr_X = X_sub.copy()
    curr_indices = np.arange(len(sub_df))

    while True:
        if len(curr_X) < k:
            return None, None, None
        model = AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average"
        )
        temp_labels = model.fit_predict(curr_X)
        counts = pd.Series(temp_labels).value_counts()
        to_remove = counts[counts < min_samples].index

        if len(to_remove) == 0:
            break

        mask_stable = ~np.isin(temp_labels, to_remove)
        curr_X = curr_X[mask_stable]
        curr_indices = curr_indices[mask_stable]

    Z = linkage(curr_X, method="average", metric="cosine")
    stable_headlines = sub_df.iloc[curr_indices]["headline"].tolist()
    return Z, stable_headlines, curr_X


def plot_hac_dendrogram_plotly(Z, leaf_labels, start_date, end_date):
    if Z is None:
        print("No stable clustering results to plot.")
        return None

    # Corrigé : dummy_X n'a plus besoin d'être de taille 300
    dummy_X = np.zeros((len(leaf_labels), 1))

    fig = ff.create_dendrogram(
        dummy_X,
        orientation="bottom",
        labels=leaf_labels,
        linkagefun=lambda x: Z,
    )
    fig.update_layout(
        title=f"<b>Stable HAC Dendrogram</b><br><sup>Period: {start_date} to {end_date}</sup>",
        width=1000,
        height=800,
        template="plotly_white",
        xaxis=dict(title="Financial News Articles (Stable Subset)"),
        yaxis=dict(title="Cosine Distance"),
    )
    return fig


######################### Outliers removal #######################


################### Extraction des clusters stables ##################
def get_stable_clusters(X_full, df_features, start_date, end_date, k, min_samples=2):
    """
    Exécute la logique de stabilité HAC et retourne les données brutes
    prêtes pour l'outlier removal.
    """
    mask = (df_features["date"] >= start_date) & (df_features["date"] <= end_date)
    X_sub = X_full[mask.values]
    sub_df = df_features.loc[mask].copy()

    curr_X = X_sub.copy()
    curr_indices = np.arange(len(sub_df))

    while True:
        if len(curr_X) < k:
            return None, None, None
        model = AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average"
        )
        temp_labels = model.fit_predict(curr_X)
        counts = pd.Series(temp_labels).value_counts()
        to_remove = counts[counts < min_samples].index

        if len(to_remove) == 0:
            break

        mask_stable = ~np.isin(temp_labels, to_remove)
        curr_X = curr_X[mask_stable]
        curr_indices = curr_indices[mask_stable]

    stable_df = sub_df.iloc[curr_indices].copy()
    stable_df["Cluster"] = temp_labels  # Ajout des labels finaux

    return curr_X, stable_df, temp_labels


################### Event Signatures (Centroïdes) ####################
def calculate_event_centroids(X, labels):
    """
    Calcule le centroïde (vecteur médian) pour chaque cluster.
    La médiane est plus robuste aux outliers extrêmes.
    """
    unique_labels = np.unique(labels)
    centroids = {}

    for label in unique_labels:
        cluster_samples = X[labels == label]
        # axis=0 : on calcule la médiane pour chaque dimension du vecteur
        event_signature = np.median(cluster_samples, axis=0)
        centroids[label] = event_signature

    return centroids


#################### Advanced Outlier Removal #######################
def remove_news_outliers_advanced(X, labels, percentile_threshold=20):
    """
    Double filtre de Carta et al. :
    1. Ambiguïté (Silhouette)
    2. Isolement (Similarité Cosinus avec la médiane du cluster)
    """
    n_samples = len(X)

    # Sécurité : la silhouette a besoin d'au moins 2 clusters
    if len(np.unique(labels)) < 2:
        return np.ones(n_samples, dtype=bool)

    # 1. Calcul de l'ambiguïté (Silhouette individuelle)
    sil_scores = silhouette_samples(X, labels, metric="cosine")

    # 2. Calcul de l'isolement (Distance par rapport au centroïde)
    centroids = calculate_event_centroids(X, labels)
    sim_scores = np.zeros(n_samples)

    for i in range(n_samples):
        cluster_label = labels[i]
        centroid = centroids[cluster_label]
        # scipy.spatial.distance.cosine retourne la distance (0 à 2).
        # La similarité est 1 - distance.
        sim_scores[i] = 1 - cosine(X[i], centroid)

    # Définition des seuils (ex: les 20% les plus mauvais)
    sil_thresh = np.percentile(sil_scores, percentile_threshold)
    sim_thresh = np.percentile(sim_scores, percentile_threshold)

    # On conserve l'article SEULEMENT s'il réussit les DEUX tests
    mask_keep = (sil_scores >= sil_thresh) & (sim_scores >= sim_thresh)

    return mask_keep



def generate_model_wordclouds(df_clean, model_name, text_column="headline"):
    """
    Génère et affiche un nuage de mots pour chaque cluster d'un modèle donné.
    """
    # S'il n'y a pas de données (ex: la logique de stabilité a échoué), on arrête
    if df_clean is None or df_clean.empty:
        print(f"Pas de données pour le modèle {model_name}.")
        return

    clusters = sorted(df_clean["Cluster"].unique())
    num_clusters = len(clusters)
    
    print(f"\nGénération des nuages de mots pour {model_name} ({num_clusters} clusters)...")

    # Création d'une grille de sous-graphiques (1 ligne, num_clusters colonnes)
    fig, axes = plt.subplots(1, num_clusters, figsize=(8 * num_clusters, 8))
    
    # Si on n'a qu'un seul cluster, axes n'est pas une liste, on le force en liste
    if num_clusters == 1:
        axes = [axes]

    # Configuration des Stopwords
    stopwords = set(STOPWORDS)
    # Vous pouvez ajouter ici du bruit financier spécifique à ignorer
    stopwords.update(["say", "says", "said", "will", "new", "year", "stock", "stocks", "market","s"])

    for ax, cluster_id in zip(axes, clusters):
        # 1. Isoler les textes du cluster spécifique
        cluster_texts = df_clean[df_clean["Cluster"] == cluster_id][text_column]
        
        plot_words = ''
        
        # 2. Votre logique de nettoyage (itération, string, lower, split)
        for val in cluster_texts:
            val = str(val)
            tokens = val.split()
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()
            plot_words += " ".join(tokens) + " "
            
        # 3. Création du WordCloud
        wordcloud = WordCloud(
            width=800, 
            height=800,
            background_color='white',
            stopwords=stopwords,
            min_font_size=10,
            colormap='cividis' # Une belle palette de couleurs au choix (viridis, magma, plasma...)
        ).generate(plot_words)
        
        # 4. Affichage dans le sous-graphique dédié
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(f"{model_name} | Cluster {cluster_id}", fontsize=24, fontweight='bold', pad=20)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
