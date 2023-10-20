from sentence_transformers import SentenceTransformer
from bertopic.representation import KeyBERTInspired
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP


def extract_topics(
        texts,
        embedder_name="all-MiniLM-L6-v2",
        n_neighbors=15,
        n_components=10,
        umap_metric='cosine',
        random_state=42,
        min_cluster_size=50,
        hdbscan_metric='euclidean'
):
    embedding_model = SentenceTransformer(embedder_name)
    embeddings = embedding_model.encode(texts)

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric=umap_metric,
        random_state=random_state
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric=hdbscan_metric,
        cluster_selection_method='eom',
        prediction_data=True
    )

    representation_model = KeyBERTInspired()

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        representation_model=representation_model,
        top_n_words=10,
        verbose=False
    )

    topics, _ = topic_model.fit_transform(texts, embeddings)

    return topics, topic_model, embeddings


def dim_reduction_texts_topics(textual_embeddings, topic_embeddings):
    umap_model = UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )

    projected_texts = umap_model.fit_transform(textual_embeddings)
    projected_topics = umap_model.transform(topic_embeddings)

    return projected_texts, projected_topics
