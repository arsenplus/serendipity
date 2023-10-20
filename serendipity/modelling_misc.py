import spacy
from tqdm.auto import tqdm
from transformers import pipeline


def extract_named_entities(topics, texts):
    nlp = spacy.load('en_core_web_sm')
    ner_topics = []
    all_ents = []

    for topic, text in tqdm(zip(topics, texts), total=len(texts)):
        doc = nlp(text)
        for ent in doc.ents:
            ner_topics.append(topic)
            all_ents.append(str(ent))

    return ner_topics, all_ents


def extract_custom_classes(candidate_labels, texts, bs=16):
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
