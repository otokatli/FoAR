# FROM continuumio/miniconda3
FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

LABEL maintainer="ozan.tokatli@ukaea.uk"

ARG DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda

# Change shell to bash
SHELL ["/bin/bash", "-c"]

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    libopenblas-dev && \
    rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

# Get the python dependency file
COPY environment.yml .

# Install Miniconda and create the new environment
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    /bin/bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    /opt/conda/bin/conda init && \
    /opt/conda/bin/conda update -n base -c defaults conda && \
    /opt/conda/bin/conda env create -f environment.yml && \
    git clone git@github.com:chenxi-wang/MinkowskiEngine.git
    # cd MinkowskiEngine && \
    # python setup.py install --blas_include_dirs=${CUDA_HOME}/include --blas_library_dirs=${CUDA_HOME}/lib64 --blas=openblas


# Set up the shell for Conda
ENV PATH="/opt/conda/bin:$PATH"
SHELL ["/bin/bash", "--login", "-c"]

RUN echo "conda activate foar" >> ~/.bashrc

# Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "foar", "/bin/bash", "-c"]