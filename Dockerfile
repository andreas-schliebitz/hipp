FROM python:3.8

ENV TZ=Europe/Berlin

ARG POETRY_VERSION="1.8.2"

RUN apt update
RUN apt install -y libgl1

RUN curl -sSL https://install.python-poetry.org \
    | python - --version "${POETRY_VERSION}"
RUN mv /root/.local/bin/poetry /usr/local/bin/poetry

WORKDIR /workspace

COPY pyproject.toml example.py README.md ./
COPY hipp hipp

RUN poetry config virtualenvs.in-project true
RUN poetry lock
RUN poetry install

ENV PATH=/workspace/.venv/bin:$PATH

VOLUME [ "/workspace/data" ]

ENTRYPOINT [ "python", "example.py" ]
CMD [ "$@" ]