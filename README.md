# serendipity
Open-source tool for EDA-analysis of textual corpora. Built with BERTopic, transformers, and spaCy.

This tool would be useful for social science researchers, data journalists, busiseness analysts and PhD students, who regularly need to gain insights from text data, but have little experience with tuning NLP pipelines.

This repo contains a minimal working prototype, demonstrating the potential of the technology. It has been developed during AI Talent Hub Hackathon organized by ITMO University in September, 2023.

## Installation
- clone this repo to your local machine
- make sure you have Docker and docker-compose pre-installed
- run the following command:
```
docker-compose up -d --build
```
- then access the interface at http://localhost:9999

## TO-DO:
- implement coreference resolution and zero-shot ner for more accuracy/custom entities extraction
- improve the UX and data viz
- implement hyperparams auto-tuning with bayesian optimization techniques
- add sentiment analysis feature
