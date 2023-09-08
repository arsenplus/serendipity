# serendipity

![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)<br/>
Open-source tool for EDA-analysis of textual corpora. Built with BERTopic, transformers, and spaCy.

This tool would be useful for social science researchers, data journalists, busiseness analysts and PhD students, who regularly need to gain insights from text data, but have little experience with tuning NLP pipelines.

This repo contains a minimal working prototype, demonstrating the potential of the technology. It has been developed during AI Talent Hub Hackathon organized by the ITMO University in September, 2023.

## Installation
- clone this repo to your local machine
- make sure you have Docker and docker-compose pre-installed
- run the following command:
```
docker-compose up -d --build
```

<div id="header" align="center"> <img src=https://github.com/arsenplus/serendipity/blob/main/show(1).gif width="1000"/>
  </div>
- then access the interface at http://localhost:9999
<div id="header" align="center"> <img src=https://github.com/arsenplus/serendipity/blob/main/show2(1).gif width="1000"/>
  </div>
## Algorithm
- topic modelling pipeline (embeddings -> dimensionality reduction -> clustering -> representation) extracts topics
- NER model extracts named entities
- zero-shot classificator categorizes the texts into one or more custom classes
- corpus statistics (n-gram counts) are gathered per the whole corpus and each topic separately
- Interactive DataViz dashboards are built on top of the preceding steps

## TO-DO:
- implement coreference resolution and zero-shot ner for more accuracy/custom entities extraction
- implement UI and improve data viz
- implement hyperparams auto-tuning with bayesian optimization techniques
- add sentiment analysis feature
- compress classification model for speedup
