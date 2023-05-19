FROM python:3.9-slim

ENV PYTHONUNBUFFERED 1

WORKDIR /app

ADD . /app

RUN apt-get update && apt-get install wget -y
RUN apt install git -y
RUN pip install -r requirements.txt
