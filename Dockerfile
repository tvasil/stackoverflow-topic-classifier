# Inspired by https://yu-ishikawa.medium.com/machine-learning-as-a-microservice-in-python-16ba4b9ea4ee
FROM python:3.8-slim
WORKDIR /stackoverflow

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        pkg-config \
        rsync \
        unzip \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install local packages
COPY /core ./core/
COPY /prediction ./prediction/
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --upgrade setuptools\
    && python3 -m pip install ./core/ ./prediction/

# Ensure NLTK corpus is also installed
RUN python3 -m nltk.downloader punkt stopwords

# Install gRPC related packages
RUN python3 -m pip install grpcio grpcio-reflection

EXPOSE 50051
COPY nostradamus.service /stackoverflow/
RUN mkdir "/models"

CMD ["python", "nostradamus_server.py"]
