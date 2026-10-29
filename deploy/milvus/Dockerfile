# Reuse the AGFS binary and runtime dependencies from the existing oGMemory image.
ARG OGMEM_BASE_IMAGE=swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/ogmemory:1013
FROM ${OGMEM_BASE_IMAGE}
USER root
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ARG PYMILVUS_VERSION=2.6.17
ARG INSTALL_TEST_DEPS=false
RUN pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" "pymilvus==${PYMILVUS_VERSION}" && \
    if [ "$INSTALL_TEST_DEPS" = "true" ]; then \
      pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" pytest; \
    fi
COPY core/ /opt/ogmem/core/
COPY providers/ /opt/ogmem/providers/
COPY features/ /opt/ogmem/features/
COPY retrieval/ /opt/ogmem/retrieval/
COPY extraction/ /opt/ogmem/extraction/
COPY commit/ /opt/ogmem/commit/
COPY index/ /opt/ogmem/index/
COPY service/ /opt/ogmem/service/
COPY fs/ /opt/ogmem/fs/
COPY server/ /opt/ogmem/server/
COPY session/ /opt/ogmem/session/
COPY cli/ /opt/ogmem/cli/
COPY scripts/ /opt/ogmem/scripts/
COPY skills/ /opt/ogmem/skills/
COPY scene/ /opt/ogmem/scene/
COPY integrations/ /opt/ogmem/integrations/
COPY perf/ /opt/ogmem/perf/
COPY m3/ /opt/ogmem/m3/
WORKDIR /opt/ogmem
ENV PYTHONPATH=/opt/ogmem
USER ogmem
ENTRYPOINT ["python3", "/opt/ogmem/scripts/standalone_runtime.py"]
