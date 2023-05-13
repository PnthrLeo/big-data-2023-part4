FROM python:3.9-slim

ENV PYTHONUNBUFFERED 1

WORKDIR /app

ADD . /app

RUN apt-get update && apt-get install wget -y
RUN wget https://github.com/mozilla/sops/releases/download/v3.7.3/sops-v3.7.3.linux.amd64 && \
    mv sops-v3.7.3.linux.amd64 /usr/local/bin/sops && \
    chmod +x /usr/local/bin/sops
RUN apt install gnupg2 -y
RUN apt install git -y
RUN pip install -r requirements.txt
