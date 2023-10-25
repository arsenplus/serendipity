import math
import torch
import wikipedia
from tqdm.auto import tqdm


def extract_relations_from_model_output(text):
    relations = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text_replaced.split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        relations.append({
            'head': subject.strip(),
            'type': relation.strip(),
            'tail': object_.strip()
        })
    return relations


class KB():
    def __init__(self):
        self.entities = {}
        self.relations = []


    def merge_with_kb(self, kb2, text):
        for rel in kb2.relations:
            self.add_relation(rel, text)


    def are_relations_equal(self, rel1, rel2):
        return all(rel1[attr] == rel2[attr] for attr in ["head", "type", "tail"])


    def exists_relation(self, rel1):
        return any(self.are_relations_equal(rel1, rel2) for rel2 in self.relations)


    def merge_relations(self, rel2):
        rel1 = [rel for rel in self.relations if self.are_relations_equal(rel2, rel)][0]
        text = list(rel2["meta"].keys())[0]

        if text not in rel1["meta"]:
            rel1["meta"][text] = rel2["meta"][text]
        else:
            spans_to_add = [span for span in rel2["meta"][text]["spans"] if span not in rel1["meta"][text]["spans"]]
            rel1["meta"][text]["spans"] += spans_to_add


    def get_wikipedia_data(self, candidate_entity):
        try:
            page = wikipedia.page(candidate_entity, auto_suggest=False)
            entity_data = {
                "title": page.title,
                "url": page.url,
                "summary": page.summary
            }
            return entity_data
        except:
            return None


    def add_entity(self, ent):
        topic_label = ent["topic_label"][0]
        if ent["title"] not in self.entities:
            self.entities[ent["title"]] = {k:v for k,v in ent.items() if k != "title"}
        else:
            self.entities[ent["title"]]["topic_label"].append(topic_label)


    def add_relation(self, rel, text):
        candidate_entities = [rel["head"], rel["tail"]]
        entities = [self.get_wikipedia_data(ent) for ent in candidate_entities]

        if any(ent is None for ent in entities):
            return

        topic_label = rel["meta"][text]["topic_label"]
        for ent in entities:
            ent["topic_label"] = [topic_label]

        for ent in entities:
            self.add_entity(ent)

        rel["head"] = entities[0]["title"]
        rel["tail"] = entities[1]["title"]

        if not self.exists_relation(rel):
            self.relations.append(rel)
        else:
            self.merge_relations(rel)



def from_text_to_kb(
        text,
        topic_label,
        model,
        tokenizer,
        span_length=128
        ):

    inputs = tokenizer([text], return_tensors="pt")

    num_tokens = len(inputs["input_ids"][0])
    num_spans = math.ceil(num_tokens / span_length)
    overlap = math.ceil((num_spans * span_length - num_tokens) / max(num_spans - 1, 1))

    spans_boundaries = []
    start = 0

    for i in range(num_spans):
        spans_boundaries.append([start + span_length * i, start + span_length * (i + 1)])
        start -= overlap

    tensor_ids = [
        inputs["input_ids"][0][boundary[0]:boundary[1]] for boundary in spans_boundaries
        ]
    tensor_masks = [
        inputs["attention_mask"][0][boundary[0]:boundary[1]] for boundary in spans_boundaries
        ]
    inputs = {
        "input_ids": torch.stack(tensor_ids),
        "attention_mask": torch.stack(tensor_masks)
        }
    num_return_sequences = 3

    gen_kwargs = {
        "max_length": 256,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": num_return_sequences
        }

    generated_tokens = model.generate(
        **inputs,
        **gen_kwargs,
        )

    decoded_preds = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=False
        )

    kb = KB()

    idx = 0
    for sentence_pred in decoded_preds:
        current_span_index = idx // num_return_sequences
        relations = extract_relations_from_model_output(sentence_pred)
        for relation in relations:
            relation["meta"] = {
                text: {
                    "spans": [spans_boundaries[current_span_index]],
                    "topic_label": topic_label
                }
            }
            kb.add_relation(relation, text)
        idx += 1

    return kb


def from_corpus_to_kb(
        corpus,
        topic_labels,
        model,
        tokenizer
        ):
    kb = KB()
    for text, label in tqdm(zip(corpus, topic_labels)):
        kb_text = from_text_to_kb(text, label, model, tokenizer)
        kb.merge_with_kb(kb_text, text)
    return kb
