FROM python:3.8

RUN apt update
RUN apt install -y --no-install-recommends libgl1

COPY src/ /src

WORKDIR /src

RUN python -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

WORKDIR /workspace
COPY example.py .

VOLUME [ "/workspace/data" ]

ENTRYPOINT [ "python3", "example.py" ]
CMD [ "$@" ]
