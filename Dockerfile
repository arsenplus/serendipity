FROM python:3.8

WORKDIR /workspace
COPY requirements.txt main.py data cache model serendipity.py ./

RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm

# EXPOSE 9999

# CMD ["python", "/workspace/main.py"]
# ENTRYPOINT ["python", "main.py"]
