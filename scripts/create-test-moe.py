#!/usr/bin/env python3
"""Create the tiny Qwen3-MoE GGUF used by paging tests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import zlib

sys.dont_write_bytecode = True

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "moe_paging"
DEFAULT_OUTPUT = FIXTURE_DIR / "qwen3moe-tiny-f32.gguf"
SEED = 20261101

sys.path.insert(0, str(ROOT / "third_party" / "llama.cpp" / "gguf-py"))
import gguf

N_VOCAB = 260
N_EMBD = 32
N_LAYER = 2
N_HEAD = 4
N_HEAD_KV = 4
N_EXPERT = 4
N_EXPERT_USED = 2
N_FF = 32
N_FF_EXP = 16
HEAD_DIM = N_EMBD // N_HEAD


def tensor_name(tensor_type: gguf.MODEL_TENSOR, layer: int | None = None) -> str:
    base = gguf.TENSOR_NAMES[tensor_type]
    if layer is not None:
        base = base.format(bid=layer)
    return base + ".weight"


def tensor_specs() -> list[tuple[str, tuple[int, ...], bool]]:
    specs = [
        (tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD), (N_VOCAB, N_EMBD), False),
        (tensor_name(gguf.MODEL_TENSOR.OUTPUT_NORM), (N_EMBD,), True),
        (tensor_name(gguf.MODEL_TENSOR.OUTPUT), (N_VOCAB, N_EMBD), False),
    ]

    for layer in range(N_LAYER):
        specs.extend(
            [
                (tensor_name(gguf.MODEL_TENSOR.ATTN_NORM, layer), (N_EMBD,), True),
                (tensor_name(gguf.MODEL_TENSOR.ATTN_Q, layer), (N_EMBD, N_EMBD), False),
                (tensor_name(gguf.MODEL_TENSOR.ATTN_K, layer), (N_EMBD, N_EMBD), False),
                (tensor_name(gguf.MODEL_TENSOR.ATTN_V, layer), (N_EMBD, N_EMBD), False),
                (tensor_name(gguf.MODEL_TENSOR.ATTN_OUT, layer), (N_EMBD, N_EMBD), False),
                (tensor_name(gguf.MODEL_TENSOR.ATTN_K_NORM, layer), (HEAD_DIM,), True),
                (tensor_name(gguf.MODEL_TENSOR.ATTN_Q_NORM, layer), (HEAD_DIM,), True),
                (tensor_name(gguf.MODEL_TENSOR.FFN_NORM, layer), (N_EMBD,), True),
                (tensor_name(gguf.MODEL_TENSOR.FFN_GATE_INP, layer), (N_EXPERT, N_EMBD), False),
                (
                    tensor_name(gguf.MODEL_TENSOR.FFN_GATE_EXP, layer),
                    (N_EXPERT, N_FF_EXP, N_EMBD),
                    False,
                ),
                (
                    tensor_name(gguf.MODEL_TENSOR.FFN_DOWN_EXP, layer),
                    (N_EXPERT, N_EMBD, N_FF_EXP),
                    False,
                ),
                (
                    tensor_name(gguf.MODEL_TENSOR.FFN_UP_EXP, layer),
                    (N_EXPERT, N_FF_EXP, N_EMBD),
                    False,
                ),
            ]
        )

    return specs


def tensor_values(name: str, shape: tuple[int, ...], is_norm: bool) -> np.ndarray:
    if is_norm:
        return np.ones(shape, dtype=np.float32)

    count = int(np.prod(shape))
    offset = zlib.crc32(name.encode("ascii")) + SEED
    values = (np.arange(count, dtype=np.int64) * 37 + offset) % 257 - 128
    return (values.astype(np.float32) / 4096.0).reshape(shape)


def add_metadata(writer: gguf.GGUFWriter) -> None:
    writer.add_name("Synthetic Qwen3-MoE paging fixture")
    writer.add_vocab_size(N_VOCAB)
    writer.add_context_length(64)
    writer.add_embedding_length(N_EMBD)
    writer.add_block_count(N_LAYER)
    writer.add_feed_forward_length(N_FF)
    writer.add_expert_feed_forward_length(N_FF_EXP)
    writer.add_head_count(N_HEAD)
    writer.add_head_count_kv(N_HEAD_KV)
    writer.add_rope_dimension_count(HEAD_DIM)
    writer.add_rope_freq_base(1000000.0)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_expert_count(N_EXPERT)
    writer.add_expert_used_count(N_EXPERT_USED)

    byte_encoder = gguf.bytes_to_unicode()
    tokens = [byte_encoder[index] for index in range(256)]
    tokens.extend(["<|endoftext|>", "<|im_start|>", "<|im_sep|>", "<|im_end|>"])
    token_types = [gguf.TokenType.NORMAL] * 256 + [gguf.TokenType.CONTROL] * 4

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen2")
    writer.add_token_list(tokens)
    writer.add_token_merges([f"{tokens[0]} {tokens[1]}"])
    writer.add_token_types(token_types)
    writer.add_bos_token_id(256)
    writer.add_eos_token_id(259)
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)


def create_fixture(output: Path) -> None:
    fixture_dir = FIXTURE_DIR.resolve()
    output = output.resolve()
    if output.parent != fixture_dir:
        raise ValueError(f"output must be directly inside {fixture_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(output, gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.QWEN3MOE])
    try:
        add_metadata(writer)
        for name, shape, is_norm in tensor_specs():
            writer.add_tensor(name, tensor_values(name, shape, is_norm))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    create_fixture(args.output)
    print(f"Wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
