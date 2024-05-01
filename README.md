# HIPP: Hyperspectral Image Preprocessing Pipelines

This repository contains the Python code used for creating hyperspectral preprocessing pipelines in the LWDA 2023 publication

> [Preprocessing Ground-Based Hyperspectral Image Data for Improving CNN-based Classification](https://ceur-ws.org/Vol-3630/LWDA2023-paper35.pdf)

by Andreas Schliebitz et al. For citing this work, please se [Citing HIPP](#citing-hipp).

## Installation

This project uses `poetry` for managing Python dependencies (see [pyproject.toml](./pyproject.toml)). As the code was only tested using Python 3.8 in conjunction with dependencies that were current at that time, a Dockerfile with `python:3.8` is provided as execution environment.

### Installing on host

This installation method is intended for users, who have Python 3.8 natively installed on their systems. As time progresses, more and more users will update to newer and therefore untested Python versions. If you don't have Python 3.8 installed, you can use the the `Dockerfile` method below.

1. Install Poetry:

    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

2. Create and activate a virtual environment:

    ```bash
    poetry shell
    ```

3. Install the requirements:

    ```bash
    poetry lock
    poetry install
    ```

### Installing inside Docker

If Python 3.8 is not natively installed on your system, you can use the provided [Dockerfile](./Dockerfile) to create and run preprocessing pipelines using HIPP _in a tested environment_:

1. Build the Docker image:

    ```bash
    docker build -t hipp .
    ```

2. Instantiate the image and run `example.py` on the example hypercube inside of [data](./data) using a dockerized environment:

    ```bash
    docker run --rm -it --name hipp -v ./data:/workspace/data hipp
    ```

In order to run your own HIPP code inside of Docker, simply change the `ENTRYPOINT` of the Dockerfile and mount your own datasets as Docker volumes.

## License

The code in this repository is released unter the Creative Commons CC BY 4.0 License. See [LICENSE](./LICENSE) for additional details.

## Citing HIPP

If you find this repository useful, please consider citing it in your work:

```text
@inproceedings{Schliebitz2023,
  title={Preprocessing Ground-Based Hyperspectral Image Data for Improving CNN-based Classification},
  author={Andreas Schliebitz and Heiko Tapken and Martin Atzm{\"u}ller},
  booktitle={Lernen, Wissen, Daten, Analysen},
  year={2023},
  url={https://ceur-ws.org/Vol-3630/LWDA2023-paper35.pdf}
}
```
