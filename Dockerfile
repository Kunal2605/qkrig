FROM amazonlinux:2023 AS qkrig_base

# System deps:
#   python3.11 + pip + devel  — runtime + headers for cartopy source build
#   git, tar, xz              — git clone + uv installer
#   gcc, geos-devel, proj-devel — required by cartopy / pyproj / pykrige
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

# Editable install of qkrig + all runtime deps into the system Python.
# --system: skip venv (the container IS the isolation boundary).
RUN uv pip install --system -e . \
    && rm -rf /root/.cache/uv /root/.cache/pip

# Outputs and KV cache live here. Mount host volumes / S3 fuse mounts to persist.
RUN mkdir -p /qkrig/exports /qkrig/usgs_hourly_retrieval_logs
VOLUME ["/qkrig/exports", "/qkrig/usgs_hourly_retrieval_logs"]

# Threading hygiene: the dispatch script parallelizes 23 hours/day; per-process
# math libs should stay single-threaded to avoid oversubscription.
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    MPLBACKEND=Agg

# Entry point: the hourly dispatch script.
# Override CMD to pass [CONFIG] [START_DATE] [END_DATE]:
#   docker run --rm \
#     -v /host/exports:/qkrig/exports \
#     qkrig:1.0.0 configs/usgsgaugekrig.yaml 2025-07-01 2025-07-01
ENTRYPOINT ["bash", "Scripts/dispatch_usgs_krig_range_subdaily.sh"]
CMD ["configs/usgsgaugekrig.yaml"]
