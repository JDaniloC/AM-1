FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

# Install Python 3.8
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get -y install --no-install-recommends python3.8 python3.8-venv python3.8-dev curl git build-essential \
    && curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3.8
    

# Set Python 3.8 as default (optional, but helpful)
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
