FROM debian:stable-slim

RUN apt-get update && \
    apt-get install -y git procps python3-pip

RUN pip install \
        transformers diffusers accelerate scipy
