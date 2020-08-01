FROM python:3-slim
MAINTAINER x0rzkov <x0rzkov@protonmail.com>

RUN apt-get update \
    && apt-get install --no-install-recommends -y gcc g++ \
    && pip3 install --upgrade pip \
    && mkdir -p /opt/service /opt/data

WORKDIR /opt/service

COPY . .

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "./server.py"]

