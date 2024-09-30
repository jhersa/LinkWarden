FROM python:3.11.0-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install -y git

RUN pip install -r requirements.txt

COPY ./train ./train/