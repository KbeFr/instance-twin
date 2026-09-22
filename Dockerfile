FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgeos-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

# flexComm
RUN --mount=type=ssh,id=custom git clone -b TestBranch-HDT-Project git@github.com:BertVanAcker/flexCommunicator.git /app/flexCommunicator
RUN pip install --no-cache-dir -e /app/flexCommunicator

# core_msgs
RUN --mount=type=ssh,id=custom git clone git@github.com:KbeFr/core-msgs.git /app/core-msgs
RUN pip install --no-cache-dir -e /app/core-msgs


# this service
COPY . /app/instance-twin
RUN pip install --no-cache-dir -e /app/instance-twin

WORKDIR /app/instance_twin

ENV PYTHONUNBUFFERED=1 \
    TWIN_TICK_HZ=10 \
    TWIN_NAMESPACE=default_ns \
    TWIN_NAME=InstanceTwin \
    MQTT_BROKER_HOST=mosquitto \
    REDIS_HOST=redis \
    TWIN_DRIVE_AGENT=true

CMD ["instance-twin"]