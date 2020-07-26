FROM python:3-slim

RUN apt-get update \
    && apt-get install --no-install-recommends -y gcc g++ \
    && pip3 install --upgrade pip \
    && mkdir -p /opt/service

WORKDIR /opt/service

COPY . .

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "./server.py"]

