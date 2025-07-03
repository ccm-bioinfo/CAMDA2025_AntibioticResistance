FROM mambaorg/micromamba:2.3.0-debian12-slim
COPY --chown=$MAMBA_USER:$MAMBA_USER env/base.yml env/rgi.yml /tmp/
RUN micromamba install -yn base -f /tmp/base.yml && \
    micromamba create -yf /tmp/rgi.yml && \
    micromamba clean -ay && \
    rm /tmp/base.yml /tmp/rgi.yml
WORKDIR /ext
