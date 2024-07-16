FROM python:3.9

RUN useradd -m -u 1000 user

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt

RUN pip install  -r requirements.txt
RUN pip install  --upgrade sentence_transformers
RUN pip install  --upgrade langchain
RUN pip install -U langchain-community
RUN pip install -U langchain-huggingface
RUN pip install flask-cors

COPY --chown=user . /app

CMD ["gunicorn", "app:app", "-b", "0.0.0.0:7860"]