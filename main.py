import nltk
import pickle
import numpy as np
import pandas as pd
import gradio as gr
from bertopic import BERTopic
from nltk.corpus import stopwords

from serendipity import (
    model_topics,
    model_ner_tokens,
    model_zero_shot_classification,
    dim_reduc_texts_topics,
    topic_bubbles,
    scatter_texts,
    word_scores,
    topic_heatmap,
    classes_per_corpus,
    classes_per_topic,
    ner_per_topic,
    n_grams_per_topic
)


def make_plot(plot_type):

    if plot_type == "topic bubbles":
        return topic_bubbles(topic_model, projected_topics, texts)
    elif plot_type == "scatter texts":
        return scatter_texts(topic_model, texts, projected_texts)
    elif plot_type == "word scores":
        return word_scores(topic_model)
    elif plot_type == "topic heatmap":
        return topic_heatmap(topic_model)
    elif plot_type == "ner per topic":
        return ner_per_topic(all_ents, ner_topics)
    elif plot_type == "n grams per topic":
        return n_grams_per_topic(texts, topic_model)
    elif plot_type == "classes_per_corpus":
        return classes_per_corpus(classes)
    elif plot_type == "classes per topic":
        return classes_per_topic(classes, topics)


def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths


with gr.Blocks() as demo:
        bbc = pd.read_csv('./bbc-news-data.csv', sep='\t')
        texts = bbc['content']
        y_true = bbc['category']
        # topics, topic_model, embeddings = model_topics(texts=texts)
        # ner_topics, all_ents = model_ner_tokens##(topics, texts)
        candidate_labels = list(set(y_true))
        # classes = model_zero_shot_classification(candidate_labels, texts)

        # ЗАГЛУШКА ДЛЯ ДЕМОНСТРАЦИИ
        with open('./zs-classes.pkl', 'rb') as file:
            classes = pickle.load(file)
        with open('./ner_topics.pkl', 'rb') as file:
            ner_topics = pickle.load(file)
        with open('./topics.pkl', 'rb') as file:
            topics = pickle.load(file)
        with open('./embeddings.npy', 'rb') as file:
            embeddings = np.load(file)
        with open('./all_ents.pkl', 'rb') as file:
            all_ents = pickle.load(file)

        topic_model = BERTopic.load('./topic_model')

        projected_texts, projected_topics = dim_reduc_texts_topics(embeddings, topic_model.topic_embeddings_)

        button = gr.Radio(label="Plot type",
                          choices=['topic bubbles', 'scatter texts', 'word scores',
                                   'topic heatmap', 'ner per topic', 'n grams per topic',
                                   'classes_per_corpus', 'classes per topic'], value='scatter_plot')

        plot = gr.Plot(label="Plot")
        button.change(make_plot, inputs=button, outputs=[plot])
        demo.load(make_plot, inputs=[button], outputs=[plot])


if __name__ == "__main__":
    nltk.download('stopwords')
    stopwords = set(stopwords.words('english'))

    demo.launch(server_name="0.0.0.0", server_port=9999)
