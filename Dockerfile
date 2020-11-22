# Docker container based on latest debian image
# with miniconda docker container
FROM continuumio/miniconda3

# Set working directory in docker container
WORKDIR /home/playground

# Add YML config file to docker container
ADD env_config.yml /src/env_config.yml

# Create conda environment based on added YML file:
RUN conda env create -f /src/env_config.yml
ENV PATH /opt/conda/envs/mainenv/bin:$PATH

# Activate conda environment
RUN /bin/bash -c "source activate mainenv"