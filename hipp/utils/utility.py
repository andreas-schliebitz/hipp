#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path
import shutil
from math import ceil, sqrt
from pathlib import Path
from typing import Union

import cv2
import lz4.frame
import numpy as np
import numpy.ma as ma
from numba import njit
from sklearn.covariance import ledoit_wolf
from zstandard import ZstdCompressor, ZstdDecompressor


def imshow(data: np.ndarray, title: str = "Image 1") -> None:
    cv2.imshow(title, data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def path(path: str) -> str:
    if not path:
        return ""
    return os.path.abspath(os.path.expanduser(path))


def mkdir(path: str) -> None:
    print(f'[mkdir] Created "{path}"')
    Path(path).mkdir(parents=True, exist_ok=True)


def mv(src: str, dst: str) -> None:
    print(f'[mv] Moving "{src}" to "{dst}"')
    os.rename(src, dst)


def rm(path: str, active=True) -> None:
    print(f'[rm] Removing "{path}"')
    if active:
        shutil.rmtree(path, ignore_errors=True)


def cp(src: str, dst: str) -> None:
    print(f'[cp] Copying "{src}" -> "{dst}"')
    shutil.copy(src, dst)


def is_installed(program: str) -> bool:
    return shutil.which(program) is not None


def n_jobs() -> int:
    n = os.environ["YAIT_JOBS"] if "YAIT_JOBS" in os.environ else 1
    try:
        n = int(n)
    except:
        n = 1
    return n


def invertible_cov(X: np.ndarray) -> np.ndarray:
    C = np.cov(X, rowvar=False)
    if np.linalg.cond(C) >= 1 / np.finfo(C.dtype).eps:
        print("[invertible_cov] Using Ledoit-Wolf to create invertibe cov matrix.")
        C, _ = ledoit_wolf(X)
    assert C.shape == (X.shape[1], X.shape[1])
    return C


@njit(cache=True)
def interpolate(sample: np.ndarray, where: np.ndarray) -> np.ndarray:
    assert sample.shape[0] == where.shape[0]

    if (where == False).all():
        return sample

    interpolated_sample = np.zeros_like(sample)
    for i in range(len(where)):
        interpolated_sample[i] = _interpolate_loop(i, where[i], sample, where)

    return interpolated_sample


@njit(cache=True)
def _find_neighbour(
    start: int, direction: int, sample: np.ndarray, where: np.ndarray
) -> int:
    direction = -1 if direction < 0 else 1
    while 0 <= start < len(sample):
        if not where[start]:
            return start
        start += direction
    return -1


@njit(cache=True)
def _interpolate_loop(
    i: int, to_interpolate: bool, sample: np.ndarray, where: np.ndarray
) -> np.ndarray:
    if not to_interpolate or (where == True).all():
        return sample[i]

    if 0 < i < len(sample) - 1:
        left_pos = _find_neighbour(i - 1, -1, sample, where)
        right_pos = _find_neighbour(i + 1, 1, sample, where)
    elif i == 0:
        left_pos = _find_neighbour(i + 1, 1, sample, where)
        right_pos = _find_neighbour(left_pos + 1, 1, sample, where)
    else:
        left_pos = _find_neighbour(i - 1, -1, sample, where)
        right_pos = _find_neighbour(left_pos - 1, -1, sample, where)

    if left_pos != -1 and right_pos != -1:
        return (sample[left_pos] + sample[right_pos]) / 2

    return sample[i]


@njit(cache=True)
def normalize(data: np.ndarray) -> np.ndarray:
    return (data - data.min()) / (data.max() - data.min())

def to_cube(mat: np.ndarray) -> np.ndarray:
    assert len(mat.shape) == 2

    if is_masked_cube(mat):
        mat = mat.data

    mat = mat[np.any(mat, axis=1)]
    h, d = mat.shape

    sq = int(ceil(sqrt(h)))
    mat.resize((sq, sq, d))

    if not (mat[sq - 1, :, :] != 0).any():
        mat = mat[: sq - 1, :, :]

    return ma.masked_array(
        mat, mask=~np.repeat(mat.sum(axis=-1).astype(bool), d, axis=-1), fill_value=0
    )


def is_masked_cube(data: np.ndarray) -> bool:
    return ma.isMaskedArray(data) and len(data.mask.shape) == 3


def is_normalized(data: np.ndarray) -> bool:
    return data.max() <= 1 and data.min() >= 0


def zstd(
    mode: str,
    output_filepath: str = None,
    input_filepath: str = None,
    input_data: np.ndarray = None,
    level: int = 9,
    checksum: bool = False,
    threads: int = -1,
) -> Union[bytes, None]:
    assert (input_filepath or input_data is not None) and (mode == "decompress" or mode == "compress")

    if mode == "compress":
        ctx = ZstdCompressor(level=level, write_checksum=checksum, threads=threads)
    else:
        ctx = ZstdDecompressor()

    if input_filepath:
        if output_filepath:
            # Compress/decompress input file into output file
            with open(input_file, "rb") as ih:
                with open(output_file, "wb") as oh:
                    with ctx.stream_writer(oh) as writer:
                        writer.write(ih.read())  # compresses / decompresses
        else:
            with open(input_file, "rb") as ih:
                if mode == "compress":
                    # Compress input file into byte array
                    out = ctx.compress(ih.read())
                else:
                    # Decompress input file into byte array
                    out = ctx.decompress(ih.read())
            return out
    else:
        if not isinstance(input_data, bytes):
            input_bytes = (
                input_data.tobytes()
                if isinstance(input_data, np.ndarray)
                else bytes(input_data, "utf-8")
            )
        else:
            input_bytes = input_data
        if output_filepath:
            # Compress/decompress input bytes into output file
            with open(output_file, "wb") as oh:
                with cctx.stream_writer(oh) as writer:
                    writer.write(input_bytes)  # compresses / decompresses
        else:
            if mode == "compress":
                # Compress input bytes into byte array
                out = ctx.compress(input_bytes)
            else:
                # Decompress input bytes into byte array
                out = ctx.decompress(input_bytes)
            return out


def LZ4(
    mode: str,
    output_filepath: str = None,
    input_filepath: str = None,
    input_data: np.ndarray = None,
    level: int = 0,
    checksum: bool = False,
) -> Union[bytes, None]:
    assert (input_filepath or input_data is not None) and (mode == "decompress" or mode == "compress")

    if input_filepath:
        if output_filepath:
            if mode == "compress":
                # Compress input file into output file
                with open(input_filepath, "rb") as ih:
                    with lz4.frame.open(
                        output_filepath,
                        mode="wb",
                        compression_level=level,
                        content_checksum=checksum,
                    ) as oh:  # compresses
                        oh.write(ih.read())
            else:
                # Decompress input file into output file
                with lz4.frame.open(input_filepath, mode="rb") as ih:  # decompresses
                    with open(output_filepath, "wb") as oh:
                        oh.write(ih.read())
        else:
            if mode == "compress":
                # Compress input file into byte array
                with open(input_filepath, "rb") as ih:
                    return lz4.frame.compress(
                        ih.read(), compression_level=level, content_checksum=checksum
                    )
            else:
                # Decompress input file into byte array
                with lz4.frame.open(input_file, mode="rb") as ih:
                    return ih.read()
    else:
        if not isinstance(input_data, bytes):
            input_bytes = (
                input_data.tobytes()
                if isinstance(input_data, np.ndarray)
                else bytes(input_data, "utf-8")
            )
        else:
            input_bytes = input_data
        if output_filepath:
            if mode == "compress":
                # Compress input bytes into output file
                with lz4.frame.open(
                    output_filepath,
                    mode="wb",
                    compression_level=level,
                    content_checksum=checksum,
                ) as oh:  # compresses
                    oh.write(input_bytes)
            else:
                # Decompress input bytes into output file
                with open(output_filepath, "wb") as oh:
                    oh.write(lz4.frame.decompress(input_bytes))
        else:
            if mode == "compress":
                # Compress input bytes into byte array
                out = lz4.frame.compress(
                    input_bytes, compression_level=level, content_checksum=checksum
                )
            else:
                # Decompress input bytes into byte array
                out = lz4.frame.decompress(input_bytes)
            return out
