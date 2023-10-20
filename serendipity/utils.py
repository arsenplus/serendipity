import re
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
