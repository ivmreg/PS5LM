#!/usr/bin/env python3
"""Inspect pinned Qwen3-MoE GGUF metadata without reading tensor payloads."""

from __future__ import annotations

import argparse
import json
import re
import socket
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


HF_REPO = "Qwen/Qwen3-30B-A3B-GGUF"
HF_REVISION = "e4d4bafdfb96a411a163846265362aceb0b9c63a"
HF_FILENAME = "Qwen3-30B-A3B-Q4_K_M.gguf"
HF_SHA256 = "0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5"
HF_FILE_SIZE = 18_556_685_824

DEFAULT_MAX_DOWNLOAD = 32 * 1024 * 1024
MAX_ALLOWED_DOWNLOAD = DEFAULT_MAX_DOWNLOAD
DEFAULT_TIMEOUT = 120.0
MAX_TIMEOUT_SECONDS = 300.0
RANGE_QUANTUM = 64 * 1024
MAX_RANGE_SIZE = 1024 * 1024
MAX_STRING_SIZE = 1024 * 1024
MAX_ARRAY_ELEMENTS = 2_000_000
MAX_SELECTED_ARRAY_ELEMENTS = 10_000
MAX_METADATA_ENTRIES = 10_000
MAX_TENSORS = 10_000
MAX_DIMS = 4
MIN_METADATA_ENTRY_BYTES = 14
MIN_TENSOR_ENTRY_BYTES = 33
U64_MAX = (1 << 64) - 1

# Values are (elements per quantization block, bytes per block), from the
# pinned gguf-py/gguf/constants.py GGML_QUANT_SIZES table.
GGML_TYPE_TRAITS = {
    0: (1, 4),    # F32
    1: (1, 2),    # F16
    2: (32, 18),  # Q4_0
    3: (32, 20),  # Q4_1
    6: (32, 22),  # Q5_0
    7: (32, 24),  # Q5_1
    8: (32, 34),  # Q8_0
    9: (32, 36),  # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    19: (256, 50),  # IQ1_S
    20: (32, 18),  # IQ4_NL
    21: (256, 110),  # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136),  # IQ4_XS
    24: (1, 1),  # I8
    25: (1, 2),  # I16
    26: (1, 4),  # I32
    27: (1, 8),  # I64
    28: (1, 8),  # F64
    29: (256, 56),  # IQ1_M
    30: (1, 2),  # BF16
    34: (256, 54),  # TQ1_0
    35: (256, 66),  # TQ2_0
    39: (32, 17),  # MXFP4
    40: (64, 36),  # NVFP4
    41: (128, 18),  # Q1_0
    42: (64, 18),  # Q2_0
}

GGUF_META_FIXED = {
    0: (1, "B"),  # UINT8
    1: (1, "b"),  # INT8
    2: (2, "H"),  # UINT16
    3: (2, "h"),  # INT16
    4: (4, "I"),  # UINT32
    5: (4, "i"),  # INT32
    6: (4, "f"),  # FLOAT32
    7: (1, "?"),  # BOOL
    10: (8, "Q"),  # UINT64
    11: (8, "q"),  # INT64
    12: (8, "d"),  # FLOAT64
}
GGUF_META_STRING = 8
GGUF_META_ARRAY = 9

SELECTED_METADATA_KEYS = {
    "general.architecture",
    "general.alignment",
    "qwen3moe.block_count",
    "qwen3moe.embedding_length",
    "qwen3moe.expert_count",
    "qwen3moe.expert_used_count",
    "qwen3moe.expert_feed_forward_length",
}


class AuditError(Exception):
    """Raised when a GGUF file or range response fails validation."""


def checked_add(left: int, right: int, label: str) -> int:
    if left < 0 or right < 0 or left > U64_MAX - right:
        raise AuditError(f"{label} overflows uint64")
    return left + right


def checked_mul(left: int, right: int, label: str) -> int:
    if left < 0 or right < 0 or (left and right > U64_MAX // left):
        raise AuditError(f"{label} overflows uint64")
    return left * right


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise AuditError(f"invalid GGUF alignment: {alignment}")
    return checked_add(value, alignment - 1, "alignment") & -alignment


class DownloadBudget:
    def __init__(self, limit: int):
        if limit <= 0:
            raise AuditError("download limit must be positive")
        self.limit = limit
        self.downloaded = 0
        self.range_downloaded = 0

    @property
    def remaining(self) -> int:
        return self.limit - self.downloaded

    def charge(self, count: int, is_range: bool = False) -> None:
        if count < 0 or count > self.remaining:
            raise AuditError(f"download byte limit exceeded ({self.limit} bytes)")
        self.downloaded += count
        if is_range:
            self.range_downloaded += count


def remaining_time(deadline: float) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise AuditError("network deadline exceeded")
    return left


def _set_read_timeout(response: Any, timeout: float) -> None:
    stream = getattr(response, "fp", None)
    raw = getattr(stream, "raw", None)
    sock = getattr(raw, "_sock", None)
    if sock is not None:
        sock.settimeout(timeout)


def read_http_body(response: Any, length: int, deadline: float, budget: DownloadBudget, is_range: bool = False) -> bytes:
    if length < 0 or length > budget.remaining:
        raise AuditError(f"response body exceeds remaining {budget.remaining}-byte download budget")

    chunks = []
    received = 0
    while received < length:
        timeout = remaining_time(deadline)
        _set_read_timeout(response, timeout)
        try:
            chunk = response.read1(min(64 * 1024, length - received))
        except (OSError, socket.timeout, TimeoutError) as error:
            raise AuditError(f"HTTP body read failed: {error}") from error
        if not chunk:
            break
        budget.charge(len(chunk), is_range=is_range)
        chunks.append(chunk)
        received += len(chunk)

    if received != length:
        raise AuditError(f"short HTTP body: received {received}, expected {length} bytes")
    return b"".join(chunks)


def open_url(request: urllib.request.Request, deadline: float):
    try:
        return urllib.request.urlopen(request, timeout=remaining_time(deadline))
    except (OSError, urllib.error.URLError, socket.timeout, TimeoutError) as error:
        raise AuditError(f"HTTP request failed: {error}") from error


def content_length(response: Any) -> int:
    raw = response.headers.get("Content-Length")
    if raw is None:
        raise AuditError("HTTP response has no Content-Length")
    try:
        value = int(raw)
    except ValueError as error:
        raise AuditError(f"invalid Content-Length: {raw!r}") from error
    if value < 0:
        raise AuditError(f"invalid Content-Length: {raw!r}")
    encoding = response.headers.get("Content-Encoding", "identity").lower()
    if encoding not in ("", "identity"):
        raise AuditError(f"unexpected HTTP content encoding: {encoding}")
    return value


def fetch_hf_metadata(budget: DownloadBudget, deadline: float) -> dict[str, Any]:
    url = f"https://huggingface.co/api/models/{HF_REPO}/revision/{HF_REVISION}?blobs=true"
    request = urllib.request.Request(
        url,
        headers={"Accept-Encoding": "identity", "User-Agent": "PS5LM-GGUF-header-audit/1.0"},
    )
    with open_url(request, deadline) as response:
        if response.status != 200:
            raise AuditError(f"HF metadata API returned HTTP {response.status}")
        length = content_length(response)
        body = read_http_body(response, length, deadline, budget)

    try:
        metadata = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"invalid HF metadata JSON: {error}") from error

    if metadata.get("sha") != HF_REVISION:
        raise AuditError(f"HF revision mismatch: expected {HF_REVISION}, got {metadata.get('sha')!r}")
    sibling = next(
        (item for item in metadata.get("siblings", []) if item.get("rfilename") == HF_FILENAME),
        None,
    )
    if sibling is None:
        raise AuditError(f"HF revision does not contain {HF_FILENAME}")
    lfs = sibling.get("lfs") or {}
    if sibling.get("size") != HF_FILE_SIZE or lfs.get("sha256") != HF_SHA256:
        raise AuditError(
            "HF artifact identity changed: "
            f"size={sibling.get('size')!r}, sha256={lfs.get('sha256')!r}"
        )
    return {
        "repo": metadata.get("id", HF_REPO),
        "revision": metadata["sha"],
        "filename": HF_FILENAME,
        "size_bytes": sibling["size"],
        "lfs_sha256": lfs["sha256"],
        "metadata_api_url": url,
    }


class LocalFileSource:
    def __init__(self, path: Path):
        self.path = path
        self.file_size = path.stat().st_size
        self._file = path.open("rb")

    def read_at(self, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0 or checked_add(offset, length, "local read") > self.file_size:
            raise AuditError("read extends beyond local file")
        self._file.seek(offset)
        result = self._file.read(length)
        if len(result) != length:
            raise AuditError(f"short local read at offset {offset}")
        return result

    def close(self) -> None:
        self._file.close()


class HTTPRangeSource:
    def __init__(
        self,
        url: str,
        file_size: int,
        budget: DownloadBudget,
        deadline: float,
        quantum: int = RANGE_QUANTUM,
        max_range_size: int = MAX_RANGE_SIZE,
    ):
        self.url = url
        self.file_size = file_size
        self.budget = budget
        self.deadline = deadline
        self.quantum = quantum
        self.max_range_size = max_range_size
        self.data = bytearray()
        self.tail_min_bytes = 0
        self.ranges: list[list[int]] = []
        if file_size <= 0:
            raise AuditError("remote file size must be positive")
        if quantum <= 0 or quantum & (quantum - 1):
            raise AuditError("range quantum must be a power of two")
        if max_range_size < quantum:
            raise AuditError("maximum range size must be at least one range quantum")

    def _fetch(self, start: int, end_exclusive: int) -> None:
        if start != len(self.data) or end_exclusive <= start or end_exclusive > self.file_size:
            raise AuditError("internal range request is not a contiguous file prefix")
        length = end_exclusive - start
        if length > self.budget.remaining:
            raise AuditError(f"header exceeds the {self.budget.limit}-byte download limit")
        request = urllib.request.Request(
            self.url,
            headers={
                "Accept-Encoding": "identity",
                "Range": f"bytes={start}-{end_exclusive - 1}",
                "User-Agent": "PS5LM-GGUF-header-audit/1.0",
            },
        )
        with open_url(request, self.deadline) as response:
            if response.status != 206:
                raise AuditError(f"GGUF range request returned HTTP {response.status}, expected 206")
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            if match is None:
                raise AuditError(f"invalid Content-Range: {content_range!r}")
            got_start, got_end, got_total = map(int, match.groups())
            if (got_start, got_end, got_total) != (start, end_exclusive - 1, self.file_size):
                raise AuditError(
                    "Content-Range mismatch: "
                    f"expected bytes {start}-{end_exclusive - 1}/{self.file_size}, got {content_range!r}"
                )
            length_header = content_length(response)
            if length_header != length:
                raise AuditError(f"range Content-Length mismatch: expected {length}, got {length_header}")
            self.data.extend(read_http_body(response, length, self.deadline, self.budget, is_range=True))
            self.ranges.append([start, end_exclusive - 1])

    def _ensure(self, end_exclusive: int) -> None:
        if end_exclusive < 0 or end_exclusive > self.file_size:
            raise AuditError("requested prefix extends beyond remote file")
        while len(self.data) < end_exclusive:
            needed_end = end_exclusive
            if not self.data and needed_end <= 24:
                rounded_end = needed_end
            else:
                rounded_end = ((needed_end + self.quantum - 1) // self.quantum) * self.quantum
            safe_end = min(
                self.file_size,
                checked_add(needed_end, self.tail_min_bytes, "minimum remaining GGUF header"),
            )
            rounded_end = min(rounded_end, safe_end)
            request_end = min(rounded_end, len(self.data) + self.max_range_size, self.file_size)
            budget_end = len(self.data) + self.budget.remaining
            if request_end > budget_end:
                if needed_end > budget_end:
                    raise AuditError(f"header exceeds the {self.budget.limit}-byte download limit")
                request_end = needed_end
            if request_end <= len(self.data):
                raise AuditError("download limit prevents another range request")
            self._fetch(len(self.data), request_end)

    def set_tail_min_bytes(self, value: int) -> None:
        if value < 0:
            raise AuditError("negative minimum remaining header size")
        self.tail_min_bytes = value

    def read_at(self, offset: int, length: int) -> bytes:
        end = checked_add(offset, length, "remote read")
        self._ensure(end)
        return bytes(self.data[offset:end])

    @property
    def range_bytes(self) -> int:
        return len(self.data)


class Cursor:
    def __init__(self, source: Any):
        self.source = source
        self.offset = 0

    def read(self, length: int) -> bytes:
        end = checked_add(self.offset, length, "GGUF header cursor")
        if end > self.source.file_size:
            raise AuditError("truncated GGUF header")
        result = self.source.read_at(self.offset, length)
        if len(result) != length:
            raise AuditError("short source read")
        self.offset = end
        return result

    def skip(self, length: int) -> None:
        end = checked_add(self.offset, length, "GGUF header skip")
        if end > self.source.file_size:
            raise AuditError("GGUF metadata extends beyond file size")
        self.offset = end

    def uint8(self) -> int:
        return struct.unpack("<B", self.read(1))[0]

    def uint32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def uint64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        length = self.uint64()
        if length > MAX_STRING_SIZE:
            raise AuditError(f"GGUF string length {length} exceeds {MAX_STRING_SIZE}")
        try:
            return self.read(length).decode("utf-8")
        except UnicodeDecodeError as error:
            raise AuditError(f"invalid UTF-8 in GGUF string: {error}") from error


def metadata_scalar(cursor: Cursor, value_type: int) -> Any:
    if value_type == GGUF_META_STRING:
        return cursor.string()
    fixed = GGUF_META_FIXED.get(value_type)
    if fixed is None:
        raise AuditError(f"unsupported GGUF metadata value type {value_type}")
    length, fmt = fixed
    value = struct.unpack("<" + fmt, cursor.read(length))[0]
    if value_type == 7 and value not in (False, True):
        raise AuditError("invalid GGUF boolean value")
    return value


def set_tail_min(cursor: Cursor, value: int) -> None:
    if hasattr(cursor.source, "set_tail_min_bytes"):
        cursor.source.set_tail_min_bytes(value)


def metadata_array(cursor: Cursor, value_key: str, keep: bool, future_tail: int) -> list[Any] | None:
    element_type = cursor.uint32()
    count = cursor.uint64()
    if count > MAX_ARRAY_ELEMENTS:
        raise AuditError(f"GGUF array count {count} exceeds {MAX_ARRAY_ELEMENTS}")

    if element_type == GGUF_META_STRING:
        if keep:
            raise AuditError(f"string array {value_key} is not a supported selected value")
        for index in range(count):
            remaining = count - index - 1
            set_tail_min(cursor, future_tail + remaining * 8)
            length = cursor.uint64()
            if length > MAX_STRING_SIZE:
                raise AuditError(f"GGUF array string length {length} exceeds {MAX_STRING_SIZE}")
            set_tail_min(cursor, future_tail + length + remaining * 8)
            cursor.skip(length)
        return None

    fixed = GGUF_META_FIXED.get(element_type)
    if fixed is None:
        raise AuditError(f"unsupported GGUF array element type {element_type}")
    element_size, fmt = fixed
    total_bytes = checked_mul(count, element_size, f"GGUF array {value_key}")
    if not keep:
        cursor.skip(total_bytes)
        return None
    if count > MAX_SELECTED_ARRAY_ELEMENTS:
        raise AuditError(f"selected GGUF array {value_key} is too large to retain")
    values = []
    for index in range(count):
        set_tail_min(cursor, future_tail + (count - index - 1) * element_size)
        values.append(metadata_scalar(cursor, element_type))
    return values


def metadata_value(cursor: Cursor, key: str, value_type: int, future_tail: int) -> Any:
    if value_type == GGUF_META_ARRAY:
        return metadata_array(cursor, key, key in SELECTED_METADATA_KEYS, future_tail)
    return metadata_scalar(cursor, value_type)


def tensor_nbytes(dims: list[int], type_id: int) -> tuple[int, int, int]:
    traits = GGML_TYPE_TRAITS.get(type_id)
    if traits is None:
        raise AuditError(f"unknown GGML tensor type {type_id}; refusing to estimate size")
    block_size, type_size = traits
    if not dims or any(dim == 0 for dim in dims):
        raise AuditError("GGUF tensor dimensions must be non-zero")
    if dims[0] % block_size:
        raise AuditError(
            f"tensor row width {dims[0]} is not divisible by GGML block size {block_size}"
        )
    rows = 1
    for dim in dims[1:]:
        rows = checked_mul(rows, dim, "tensor row count")
    row_bytes = checked_mul(dims[0] // block_size, type_size, "tensor row bytes")
    return checked_mul(rows, row_bytes, "tensor bytes"), block_size, type_size


def scalar_metadata(metadata: dict[str, Any], key: str) -> int:
    value = metadata.get(key)
    if isinstance(value, list):
        if not value or any(item != value[0] for item in value):
            raise AuditError(f"{key} must be a scalar or uniform array for this Qwen3-MoE audit")
        value = value[0]
    if not isinstance(value, int) or isinstance(value, bool):
        raise AuditError(f"required integer metadata is missing or invalid: {key}")
    return value


def parse_gguf(source: Any, source_info: dict[str, Any], range_quantum: int | None = None) -> dict[str, Any]:
    cursor = Cursor(source)
    fixed_header = cursor.read(24)
    if fixed_header[:4] != b"GGUF":
        raise AuditError("invalid GGUF magic or byte order")
    version, tensor_count, metadata_count = struct.unpack("<IQQ", fixed_header[4:])
    if version not in (2, 3):
        raise AuditError(f"unsupported GGUF version {version}")
    if tensor_count > MAX_TENSORS:
        raise AuditError(f"tensor count {tensor_count} exceeds {MAX_TENSORS}")
    if metadata_count > MAX_METADATA_ENTRIES:
        raise AuditError(f"metadata entry count {metadata_count} exceeds {MAX_METADATA_ENTRIES}")
    if hasattr(source, "set_tail_min_bytes"):
        source.set_tail_min_bytes(
            metadata_count * MIN_METADATA_ENTRY_BYTES + tensor_count * MIN_TENSOR_ENTRY_BYTES
        )

    metadata: dict[str, Any] = {}
    seen_keys = set()
    for index in range(metadata_count):
        future_tail = (
            (metadata_count - index - 1) * MIN_METADATA_ENTRY_BYTES
            + tensor_count * MIN_TENSOR_ENTRY_BYTES
        )
        if hasattr(source, "set_tail_min_bytes"):
            source.set_tail_min_bytes(future_tail)
        key = cursor.string()
        if not key:
            raise AuditError("GGUF metadata key cannot be empty")
        if key in seen_keys:
            raise AuditError(f"duplicate GGUF metadata key {key!r}")
        seen_keys.add(key)
        value_type = cursor.uint32()
        value = metadata_value(cursor, key, value_type, future_tail)
        if key in SELECTED_METADATA_KEYS:
            metadata[key] = value

    architecture = metadata.get("general.architecture")
    if architecture != "qwen3moe":
        raise AuditError(f"expected qwen3moe architecture, got {architecture!r}")
    alignment = metadata.get("general.alignment", 32)
    if not isinstance(alignment, int) or isinstance(alignment, bool):
        raise AuditError("general.alignment is not an integer")
    if alignment <= 0 or alignment > 1024 * 1024 or alignment & (alignment - 1):
        raise AuditError(f"invalid GGUF alignment {alignment}")

    n_layer = scalar_metadata(metadata, "qwen3moe.block_count")
    n_embd = scalar_metadata(metadata, "qwen3moe.embedding_length")
    n_expert = scalar_metadata(metadata, "qwen3moe.expert_count")
    n_expert_used = scalar_metadata(metadata, "qwen3moe.expert_used_count")
    n_ff_exp_value = metadata.get("qwen3moe.expert_feed_forward_length")
    if n_layer <= 0 or n_embd <= 0 or n_expert <= 0 or not 0 < n_expert_used <= n_expert:
        raise AuditError("invalid Qwen3-MoE architecture dimensions or expert counts")
    n_ff_exp = None
    if n_ff_exp_value is not None:
        n_ff_exp = scalar_metadata(metadata, "qwen3moe.expert_feed_forward_length")
        if n_ff_exp <= 0:
            raise AuditError("qwen3moe.expert_feed_forward_length must be positive")

    directory_start = cursor.offset
    tensors = []
    names = set()
    for index in range(tensor_count):
        if hasattr(source, "set_tail_min_bytes"):
            source.set_tail_min_bytes((tensor_count - index - 1) * MIN_TENSOR_ENTRY_BYTES)
        name = cursor.string()
        if not name:
            raise AuditError("GGUF tensor name cannot be empty")
        if name in names:
            raise AuditError(f"duplicate GGUF tensor name {name!r}")
        names.add(name)
        ndim = cursor.uint32()
        if ndim < 1 or ndim > MAX_DIMS:
            raise AuditError(f"tensor {name!r} has invalid dimension count {ndim}")
        dims = [cursor.uint64() for _ in range(ndim)]
        type_id = cursor.uint32()
        relative_offset = cursor.uint64()
        size, block_size, type_size = tensor_nbytes(dims, type_id)
        tensors.append(
            {
                "name": name,
                "dims": dims,
                "type_id": type_id,
                "type": TYPE_NAMES[type_id],
                "block_size": block_size,
                "bytes_per_block": type_size,
                "nbytes": size,
                "relative_offset": relative_offset,
            }
        )

    directory_end = cursor.offset
    data_start = align_up(directory_end, alignment)
    if data_start > source.file_size:
        raise AuditError("tensor data start extends beyond file size")

    ranges = []
    for tensor in tensors:
        start = checked_add(data_start, tensor["relative_offset"], "absolute tensor offset")
        end = checked_add(start, tensor["nbytes"], "tensor end offset")
        if end > source.file_size:
            raise AuditError(f"tensor {tensor['name']!r} extends beyond file size")
        if tensor["relative_offset"] % alignment:
            raise AuditError(f"tensor {tensor['name']!r} offset is not GGUF-aligned")
        tensor["absolute_offset"] = start
        tensor["absolute_end"] = end
        ranges.append((start, end, tensor["name"]))
    ranges.sort()
    for (_, previous_end, previous_name), (next_start, _, next_name) in zip(ranges, ranges[1:]):
        if next_start < previous_end:
            raise AuditError(f"tensor payload ranges overlap: {previous_name!r}, {next_name!r}")

    expert_pattern = re.compile(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight\Z")
    experts_by_layer: dict[int, dict[str, dict[str, Any]]] = {}
    non_expert_bytes = 0
    for tensor in tensors:
        match = expert_pattern.fullmatch(tensor["name"])
        if match is None:
            if "_exps." in tensor["name"]:
                raise AuditError(f"unrecognized Qwen3-MoE expert tensor name {tensor['name']!r}")
            non_expert_bytes = checked_add(non_expert_bytes, tensor["nbytes"], "non-expert bytes")
            continue
        layer = int(match.group(1))
        role = match.group(2)
        if layer >= n_layer:
            raise AuditError(f"expert tensor layer {layer} exceeds block_count {n_layer}")
        dims = tensor["dims"]
        if len(dims) != 3 or dims[2] != n_expert:
            raise AuditError(f"expert tensor {tensor['name']!r} has an unexpected expert dimension")
        if role in ("gate", "up"):
            if dims[0] != n_embd:
                raise AuditError(f"expert tensor {tensor['name']!r} has an unexpected input dimension")
            expert_width = dims[1]
        else:
            if dims[1] != n_embd:
                raise AuditError(f"expert tensor {tensor['name']!r} has an unexpected output dimension")
            expert_width = dims[0]
        if n_ff_exp is not None and expert_width != n_ff_exp:
            raise AuditError(f"expert tensor {tensor['name']!r} disagrees with expert FFN metadata")
        if tensor["nbytes"] % n_expert:
            raise AuditError(f"expert tensor {tensor['name']!r} size is not divisible by expert_count")
        layer_entries = experts_by_layer.setdefault(layer, {})
        if role in layer_entries:
            raise AuditError(f"duplicate {role} expert tensor in layer {layer}")
        layer_entries[role] = tensor

    role_bytes = {"gate": 0, "up": 0, "down": 0}
    per_layer = {}
    for layer in range(n_layer):
        entries = experts_by_layer.get(layer, {})
        if set(entries) != {"gate", "up", "down"}:
            raise AuditError(f"layer {layer} does not have exactly gate/up/down expert tensors")
        gate, up, down = (entries[role] for role in ("gate", "up", "down"))
        if gate["dims"][1] != up["dims"][1] or gate["dims"][1] != down["dims"][0]:
            raise AuditError(f"layer {layer} expert FFN dimensions do not agree")
        sizes = {role: entries[role]["nbytes"] for role in ("gate", "up", "down")}
        for role, size in sizes.items():
            role_bytes[role] = checked_add(role_bytes[role], size, f"{role} expert bytes")
        layer_bytes = sum(sizes.values())
        per_layer[str(layer)] = {
            "gate_bytes": sizes["gate"],
            "up_bytes": sizes["up"],
            "down_bytes": sizes["down"],
            "total_bytes": layer_bytes,
            "per_expert_bytes": layer_bytes // n_expert,
        }

    expert_bytes = sum(role_bytes.values())
    total_tensor_bytes = checked_add(expert_bytes, non_expert_bytes, "total tensor payload bytes")
    if total_tensor_bytes > source.file_size:
        raise AuditError("tensor payload sum exceeds file size")
    role_shares = {role: role_bytes[role] / expert_bytes for role in role_bytes}
    range_bytes = getattr(source, "range_bytes", 0)
    tensor_data_downloaded = max(0, range_bytes - data_start)
    range_overread = max(0, range_bytes - directory_end)
    if range_quantum is not None and range_overread >= range_quantum:
        raise AuditError("range reader fetched more than one final quantum past tensor metadata")

    return {
        "source": source_info,
        "gguf": {
            "version": version,
            "file_size_bytes": source.file_size,
            "tensor_count": tensor_count,
            "metadata_entry_count": metadata_count,
            "directory_start_offset": directory_start,
            "directory_end_offset": directory_end,
            "tensor_data_start_offset": data_start,
            "alignment_bytes": alignment,
        },
        "architecture": {
            "name": architecture,
            "layer_count": n_layer,
            "embedding_length": n_embd,
            "expert_count": n_expert,
            "expert_used_count": n_expert_used,
            "expert_feed_forward_length": n_ff_exp_value,
        },
        "accounting": {
            "expert_tensor_count": sum(len(entries) for entries in experts_by_layer.values()),
            "expert_bytes_total": expert_bytes,
            "expert_bytes_by_role": role_bytes,
            "expert_shares_by_role": role_shares,
            "expert_bytes_per_selected_expert_per_layer": per_layer,
            "expert_bytes_per_expert_all_layers": sum(
                entry["per_expert_bytes"] for entry in per_layer.values()
            ),
            "non_expert_tensor_bytes": non_expert_bytes,
            "all_tensor_payload_bytes": total_tensor_bytes,
            "file_bytes_minus_expert_tensor_bytes": source.file_size - expert_bytes,
            "file_bytes_minus_tensor_payload_bytes": source.file_size - total_tensor_bytes,
            "expert_tensor_share_of_file": expert_bytes / source.file_size,
        },
        "transfer": {
            "api_and_range_bytes_downloaded": source_info.get("downloaded_bytes", 0),
            "range_bytes_downloaded": range_bytes,
            "range_request_count": len(getattr(source, "ranges", [])),
            "range_requests_inclusive": getattr(source, "ranges", []),
            "range_rounding_bytes_after_directory": range_overread,
            "tensor_payload_bytes_downloaded": tensor_data_downloaded,
            "range_quantum_bytes": range_quantum,
        },
        "tensors": tensors,
    }


TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4",
    41: "Q1_0", 42: "Q2_0",
}


def inspect_remote(max_download_bytes: int, timeout_seconds: float) -> dict[str, Any]:
    budget = DownloadBudget(max_download_bytes)
    deadline = time.monotonic() + timeout_seconds
    target = fetch_hf_metadata(budget, deadline)
    filename = urllib.parse.quote(HF_FILENAME, safe="")
    url = f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/{filename}"
    source = HTTPRangeSource(url, target["size_bytes"], budget, deadline)
    target["resolved_url"] = source.url
    report = parse_gguf(source, target, range_quantum=RANGE_QUANTUM)
    if report["transfer"]["tensor_payload_bytes_downloaded"] != 0:
        raise AuditError("refusing to report an audit that downloaded tensor payload bytes")
    report["transfer"]["api_and_range_bytes_downloaded"] = budget.downloaded
    return report


def inspect_local(path: Path) -> dict[str, Any]:
    source = LocalFileSource(path)
    try:
        return parse_gguf(
            source,
            {"kind": "local", "path": str(path.resolve()), "downloaded_bytes": 0},
        )
    finally:
        source.close()


def format_report(report: dict[str, Any]) -> str:
    arch = report["architecture"]
    accounting = report["accounting"]
    transfer = report["transfer"]
    lines = [
        f"Target: {report['source'].get('repo', 'local')} {report['source'].get('revision', '')}".rstrip(),
        f"Architecture: {arch['name']}, layers={arch['layer_count']}, hidden={arch['embedding_length']}, "
        f"experts={arch['expert_count']}, top_k={arch['expert_used_count']}, expert_ffn={arch['expert_feed_forward_length']}",
        f"File: {report['gguf']['file_size_bytes']:,} bytes; GGUF v{report['gguf']['version']}; "
        f"{report['gguf']['tensor_count']} tensors; data starts at {report['gguf']['tensor_data_start_offset']:,}",
        f"Expert tensors: {accounting['expert_bytes_total']:,} bytes; "
        f"gate={accounting['expert_bytes_by_role']['gate']:,}, "
        f"up={accounting['expert_bytes_by_role']['up']:,}, down={accounting['expert_bytes_by_role']['down']:,}",
        f"Non-expert tensor bytes: {accounting['non_expert_tensor_bytes']:,}; "
        f"file bytes minus expert bytes: {accounting['file_bytes_minus_expert_tensor_bytes']:,}",
        f"Range bytes: {transfer['range_bytes_downloaded']:,}; final directory rounding: "
        f"{transfer['range_rounding_bytes_after_directory']:,}; tensor payload bytes downloaded: "
        f"{transfer['tensor_payload_bytes_downloaded']:,}",
    ]
    for layer, values in accounting["expert_bytes_per_selected_expert_per_layer"].items():
        lines.append(
            f"Layer {layer}: gate={values['gate_bytes']:,}, up={values['up_bytes']:,}, "
            f"down={values['down_bytes']:,}, per expert={values['per_expert_bytes']:,} bytes"
        )
    for tensor in report["tensors"]:
        lines.append(
            f"{tensor['name']}: type={tensor['type']} dims={tensor['dims']} "
            f"bytes={tensor['nbytes']:,} offset={tensor['absolute_offset']:,}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, help="inspect a local GGUF instead of the pinned HF artifact")
    parser.add_argument("--max-download-mib", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="overall network deadline in seconds")
    parser.add_argument("--json", action="store_true", help="print the full tensor directory as JSON")
    args = parser.parse_args(argv)

    try:
        if args.max_download_mib <= 0 or args.timeout <= 0:
            raise AuditError("download limit and timeout must be positive")
        max_download_bytes = args.max_download_mib * 1024 * 1024
        if max_download_bytes > MAX_ALLOWED_DOWNLOAD:
            raise AuditError(f"download limit cannot exceed {MAX_ALLOWED_DOWNLOAD // (1024 * 1024)} MiB")
        if args.timeout > MAX_TIMEOUT_SECONDS:
            raise AuditError(f"timeout cannot exceed {MAX_TIMEOUT_SECONDS:g} seconds")
        if args.local:
            report = inspect_local(args.local)
        else:
            report = inspect_remote(max_download_bytes, args.timeout)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(format_report(report))
        return 0
    except (AuditError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
