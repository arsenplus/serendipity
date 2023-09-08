import re
import spacy
import itertools

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from bertopic import BERTopic
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer

from umap import UMAP
from tqdm.auto import tqdm
from hdbscan import HDBSCAN
from transformers import pipeline

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))


def join_docs(doc):
    return '\n\n'.join(doc)


def clean_doc(doc):
    doc = doc.lower()
    doc = re.sub('[^a-z A-Z 0-9-]+', '', doc)
    doc = " ".join([word for word in doc.split() if word not in stopwords])
    return doc


def model_topics(
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


def model_ner_tokens(topics, texts):

    nlp = spacy.load('en_core_web_sm')

    ner_topics = []
    all_ents = []

    for topic, text in tqdm(zip(topics, texts), total=len(texts)):
        doc = nlp(text)
        for ent in doc.ents:
            ner_topics.append(topic)
            all_ents.append(str(ent))

    # TO-DO: implement coreference resolution and NEs linking across docs
    return ner_topics, all_ents


def model_zero_shot_classification(candidate_labels, texts, bs=16):

    classifier = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        batch_size=bs
        )

    classes = []

    for idx in tqdm(range(0, len(texts) - bs + 1, bs)):
        batch = list(texts[idx:idx+bs])
        output = classifier(batch, candidate_labels)
        classes.extend([res['labels'][0] for res in output])

    return classes


def dim_reduc_texts_topics(textual_embeddings, topic_embeddings):

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


def topic_bubbles(
        topic_model,
        projected_topics,
        texts
        ):

    x = projected_topics[:, :1]
    y = projected_topics[:, 1:]
    topic_freq = topic_model.get_topic_freq()
    doc_info = topic_model.get_document_info(texts)
    df = topic_freq.merge(doc_info, on='Topic', how='left')
    df = df.groupby(['Topic', 'Top_n_words', 'Count', 'Name']).agg({'Probability': 'mean'}).reset_index()
    df['x'] = x
    df['y'] = y

    fig = px.scatter(
        df,
        x='x',
        y='y',
        hover_data={
            "Topic": True,
            "Top_n_words": True,
            "Count": True,
            "x": False,
            "y": False
        },
        text='Topic',
        size='Count',
        color='Name',
        size_max=100,
        template='plotly_white',
    )

    fig.update_traces(marker=dict(line=dict(width=1, color='Gray')))

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return fig


def scatter_texts(
        topic_model,
        texts,
        projected_texts
        ):

    topic_freq = topic_model.get_topic_freq()
    doc_info = topic_model.get_document_info(texts)
    df = topic_freq.merge(doc_info, on='Topic', how='left')
    x = projected_texts[:, :1]
    y = projected_texts[:, 1:]
    df['x'] = x
    df['y'] = y
    texts_c = df.groupby(['Topic']).agg({'Document': 'nunique'}).reset_index()
    texts_c = texts_c.rename(columns={'Document': 'Document_qty'})
    df = df.merge(texts_c, on='Topic', how='left')
    df.Document = df.Document.apply(lambda x: x[:100] + '...')

    fig = px.scatter(
        df,
        x='x',
        y='y',
        hover_data={
            "Topic": False,
            "Name": True,
            "Document": False,
            "Document_qty": False,
            "x": False,
            "y": False
        },
        hover_name='Document',
        color='Name',
        size_max=60,
        template='plotly_white',
    )

    fig.update_traces(marker=dict(line=dict(width=1, color='Gray')))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return fig


def word_scores(
        topic_model,
        top_n_topics=8,
        n_words=5,
        custom_labels=False,
        title="<b>Вероятности слов по темам</b>",
        width=250,
        height=250
):

    colors = itertools.cycle(["#D55E00", "#0072B2", "#CC79A7", "#E69F00", "#56B4E9", "#009E73", "#F0E442"])

    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list()[0:6])

    if isinstance(custom_labels, str):
        subplot_titles = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in topics]
        subplot_titles = ["_".join([label[0] for label in labels[:4]]) for labels in subplot_titles]
        subplot_titles = [label if len(label) < 30 else label[:27] + "..." for label in subplot_titles]
    elif topic_model.custom_labels_ is not None and custom_labels:
        subplot_titles = [topic_model.custom_labels_[topic + topic_model._outliers] for topic in topics]
    else:
        subplot_titles = [f"Тема {topic}" for topic in topics]
    columns = 4
    rows = int(np.ceil(len(topics) / columns))
    fig = make_subplots(
        rows=rows,
        cols=columns,
        shared_xaxes=False,
        horizontal_spacing=.1,
        vertical_spacing=.4 / rows if rows > 1 else 0,
        subplot_titles=subplot_titles
    )

    row = 1
    column = 1
    for topic in topics:
        words = [word + "  " for word, _ in topic_model.get_topic(topic)][:n_words][::-1]
        scores = [score for _, score in topic_model.get_topic(topic)][:n_words][::-1]

        fig.add_trace(
            go.Bar(x=scores,
                   y=words,
                   orientation='h',
                   marker_color=next(colors)),
            row=row, col=column)

        if column == columns:
            column = 1
            row += 1
        else:
            column += 1

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        title={
            'text': f"{title}",
            'x': .5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width * 4,
        height=height * rows if rows > 1 else height * 1.3,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig


def topic_heatmap(
        topic_model,
        topics=None,
        top_n_topics=None,
        n_clusters=None,
        custom_labels=False,
        title="<b>Матрица семантической близости тем</b>",
        width=800,
        height=800
):

    if topic_model.topic_embeddings_ is not None:
        embeddings = np.array(topic_model.topic_embeddings_)[topic_model._outliers:]
    else:
        embeddings = topic_model.c_tf_idf_[topic_model._outliers:]

    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]

    if top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list())

    sorted_topics = topics

    if n_clusters:
        distance_matrix = cosine_similarity(embeddings[topics])
        Z = linkage(distance_matrix, 'ward')
        clusters = fcluster(Z, t=n_clusters, criterion='maxclust')

        mapping = {cluster: [] for cluster in clusters}
        for topic, cluster in zip(topics, clusters):
            mapping[cluster].append(topic)
        mapping = [cluster for cluster in mapping.values()]
        sorted_topics = [topic for cluster in mapping for topic in cluster]

    indices = np.array([topics.index(topic) for topic in sorted_topics])
    embeddings = embeddings[indices]
    distance_matrix = cosine_similarity(embeddings)

    if isinstance(custom_labels, str):
        new_labels = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in
                      sorted_topics]
        new_labels = ["_".join([label[0] for label in labels[:4]]) for labels in new_labels]
        new_labels = [label if len(label) < 30 else label[:27] + "..." for label in new_labels]
    elif topic_model.custom_labels_ is not None and custom_labels:
        new_labels = [topic_model.custom_labels_[topic + topic_model._outliers] for topic in sorted_topics]
    else:
        new_labels = [[[str(topic), None]] + topic_model.get_topic(topic) for topic in sorted_topics]
        new_labels = ["_".join([label[0] for label in labels[:4]]) for labels in new_labels]
        new_labels = [label if len(label) < 30 else label[:27] + "..." for label in new_labels]

    fig = px.imshow(
        distance_matrix,
        labels=dict(color="Оценка близости"),
        x=new_labels,
        y=new_labels,
        color_continuous_scale='GnBu'
    )

    fig.update_layout(
        title={
            'text': f"{title}",
            'y': .95,
            'x': 0.55,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black"
            )
        },
        width=width,
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )

    fig.update_layout(showlegend=True)
    fig.update_layout(legend_title_text='Trend')

    return fig


def classes_per_corpus(classes):

    df = pd.DataFrame({'classes': classes})
    df = df.value_counts().rename_axis('classes').reset_index(name='counts')
    fig = px.bar(df, x='classes', y='counts', color='classes')

    return fig


def classes_per_topic(classes, topics,topic=1):

    df = pd.DataFrame({'classes': classes, 'topics': topics})
    df = df[df['topics'] == topic].drop(['topics'], axis=1)
    df = df.value_counts().rename_axis('classes').reset_index(name='counts')
    fig = px.bar(df, x='classes', y='counts', color='classes')

    return fig


def ner_per_topic(ents, ner_topics, topic=1):

    df = pd.DataFrame({'ents': ents, 'topics': ner_topics})
    df = df[df['topics'] == topic]
    df.drop(['topics'], inplace=True, axis=1)
    df['ents'] = df['ents'].apply(lambda x: x.strip())
    df = df.value_counts().rename_axis('entity').reset_index(name='counts').head(10)
    fig = px.bar(df, x='entity', y='counts')

    return fig


def n_grams_per_topic(texts, topic_model, topic=1, n=3):

    ngram_freq_df = pd.DataFrame()
    vectorizer = CountVectorizer(ngram_range=(n,n))
    df = topic_model.get_document_info(texts)
    df = df[df['Topic'] == topic]
    df['Document'] = df['Document'].apply(clean_doc)

    ngrams = vectorizer.fit_transform(df['Document'])
    count_values = ngrams.toarray().sum(axis=0)
    ngram_freq = pd.DataFrame(
        sorted([(count_values[i], k) for k, i in vectorizer.vocabulary_.items()],
        reverse=True),
        columns=["частота", "n-gram"]
        )

    ngram_freq_df = pd.concat([ngram_freq_df, ngram_freq])
    top_ngram = ngram_freq_df.sort_values(by='частота', ascending=False).head(10)

    fig = px.bar(
        top_ngram,
        x='частота',
        y='n-gram',
        orientation='h',
        title=f'Top-10 {n}-грамм для темы "{df.Name.iloc[0]}"'
        )

    return fig
