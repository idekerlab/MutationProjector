# Use an official Python runtime as a parent image
FROM ubuntu:24.04

# Install prerequisites
RUN apt-get update && apt-get install -y \
    wget bzip2 gcc g++ gzip tar ca-certificates build-essential cmake cython3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables for Miniconda installation
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install dependencies for Miniconda
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libc6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Download and install Miniconda

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    /bin/bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh

# Initialize Conda and update base environment
RUN conda init bash && \
    conda config --set always_yes True --set changeps1 no && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda update -n base conda

RUN mkdir /opt/MutationProjector
COPY ./ /opt/MutationProjector/


# Create cuda11 environment for NeST-VNN
RUN conda env create --file /opt/MutationProjector/conda-envs/env.yml -y

# Copy over run.sh script
COPY .dnanexus/run.sh /opt/MutationProjector/
RUN chmod a+x /opt/MutationProjector/run.sh

# Set the default shell to bash for easier Conda environment activation
SHELL ["conda", "run", "--no-capture-output", "-n", "MutationProjector", "/bin/bash", "-c"]


RUN chmod a+x /opt/MutationProjector/run.sh

ENTRYPOINT ["/opt/MutationProjector/run.sh"]

CMD ["--help"]
