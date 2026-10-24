# RAGBench in a container: `docker run --rm ragbench demo` and `docker run --rm ragbench run --preset quick ... --mock` need no API key.
# Build:  docker build -t ragbench .
# Real runs: pass the key at run time (`-e OPENAI_API_KEY`) and mount a working directory:
#   docker run --rm -e OPENAI_API_KEY -v "$PWD":/work ragbench auto --docs ./my_docs

FROM python:3.13-slim AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /src
# Only what the package build reads (setup.py copies data/demo into the wheel), so a docs or test edit does not rebuild this layer.
COPY pyproject.toml setup.py MANIFEST.in README.md LICENSE CHANGELOG.md CITATION.cff ./
COPY src ./src
COPY data/demo ./data/demo
RUN python -m venv /opt/venv && /opt/venv/bin/pip install .

FROM python:3.13-slim
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY --from=build /opt/venv /opt/venv
RUN useradd --create-home --uid 10001 ragbench && mkdir /work && chown ragbench /work
USER ragbench
WORKDIR /work
ENTRYPOINT ["ragbench"]
CMD ["--help"]
