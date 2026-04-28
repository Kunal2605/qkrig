FROM amazonlinux:2023 AS builder

RUN dnf -y install \
        git \
        tar xz \
        python3.11 python3.11-pip python3.11-devel \
        gcc gcc-c++ make \
        geos-devel proj-devel \
    && dnf clean all \
    && rm -rf /var/cache/dnf

RUN curl -LsSf https://astral.sh/uv/0.6.12/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

RUN git clone --depth 1 https://github.com/DualEarth/qkrig.git /qkrig

WORKDIR /qkrig
RUN uv pip install --system -e . \
    && rm -rf /root/.cache/uv /root/.cache/pip


FROM amazonlinux:2023 AS runtime

RUN dnf -y install \
        python3.11 \
        geos proj \
    && dnf clean all \
    && rm -rf /var/cache/dnf

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

COPY --from=builder /usr/local/lib/python3.11/site-packages   /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/lib64/python3.11/site-packages /usr/local/lib64/python3.11/site-packages
COPY --from=builder /usr/local/bin                            /usr/local/bin
COPY --from=builder /qkrig                                    /qkrig

WORKDIR /qkrig

RUN mkdir -p /qkrig/exports /qkrig/usgs_hourly_retrieval_logs
VOLUME ["/qkrig/exports", "/qkrig/usgs_hourly_retrieval_logs"]

ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    MPLBACKEND=Agg

ENTRYPOINT ["bash", "Scripts/dispatch_usgs_krig_range_subdaily.sh"]
CMD ["configs/usgsgaugekrig.yaml"]
