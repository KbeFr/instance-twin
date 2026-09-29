FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Install flexCommunicator requirements
COPY flexCommunicator/requirements.txt /app/flexCommunicator/requirements.txt
RUN pip install --no-cache-dir -r /app/flexCommunicator/requirements.txt

# Copy setup files and install local editable packages
COPY core_msgs/ /app/core_msgs
RUN pip install --no-cache-dir -e /app/core_msgs

COPY flexCommunicator/ /app/flexCommunicator
RUN pip install --no-cache-dir -e /app/flexCommunicator

# this service
COPY instance-twin /app/instance-twin
RUN pip install --no-cache-dir -e /app/instance-twin

WORKDIR /app/instance-twin

ENV PYTHONUNBUFFERED=1 \
    TWIN_TICK_HZ=10 \
    TWIN_NAMESPACE=default_ns \
    TWIN_NAME=InstanceTwin \
    MQTT_BROKER_HOST=mosquitto \
    REDIS_HOST=redis \
    TWIN_DRIVE_AGENT=true

CMD ["instance-twin"]