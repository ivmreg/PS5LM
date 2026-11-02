# Qwen3-30B-A3B Q4_K_M artifact audit

Audited 2026-11-01 from the pinned Hugging Face revision named in
[`moe-paging.md`](moe-paging.md). The audit read the HF repository metadata and
the GGUF header/tensor directory only. It did not request tensor payload bytes.

## Artifact identity

| Field | Value |
|---|---|
| Repository | `Qwen/Qwen3-30B-A3B-GGUF` |
| Revision | `e4d4bafdfb96a411a163846265362aceb0b9c63a` |
| File | `Qwen3-30B-A3B-Q4_K_M.gguf` |
| HF API file size | 18,556,685,824 bytes |
| HF LFS SHA-256 | `0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5` |
| GGUF version | 3 |
| Tensor count | 579 |
| Metadata entries | 31 |

The revision, size and LFS digest came from the [HF model API at the pinned
revision](https://huggingface.co/api/models/Qwen/Qwen3-30B-A3B-GGUF/revision/e4d4bafdfb96a411a163846265362aceb0b9c63a?blobs=true).
The API digest was not independently recomputed because the artifact payload
was not downloaded.

## GGUF findings

The file metadata reports Qwen3-MoE with 48 layers, embedding length 2,048,
expert FFN length 768, 128 experts and top-8 routing. This confirms the five
model dimensions listed as expected values in the P0 budget worksheet.

Tensor shapes below are in GGUF/GGML dimension order:

| Tensor role | Per-layer dimensions | Type | Bytes per layer | Layers |
|---|---|---:|---:|---:|
| Gate experts | `[2048, 768, 128]` | Q4_K | 113,246,208 | 48 |
| Up experts | `[2048, 768, 128]` | Q4_K | 113,246,208 | 48 |
| Down experts | `[768, 2048, 128]` | Q6_K on 24 layers; Q4_K on 24 | 165,150,720 or 113,246,208 | 48 |

The expert tensors total **17,553,162,240 bytes** (16.348 GiB): 5,435,817,984
bytes gate, 5,435,817,984 bytes up and 6,681,526,272 bytes down. Their shares
are 30.97%, 30.97% and 38.06%, respectively. The exact per-tensor offsets,
types and byte sizes are in the generated JSON report.

The 435 non-expert tensors total **997,554,176 bytes**. All 579 tensor payloads
sum to 18,550,716,416 bytes; the remaining 5,969,408 file bytes cover GGUF
metadata and alignment. File bytes minus expert tensors are 1,003,523,584.
The observed tensor types are 289 Q4_K, 49 Q6_K and 241 F32. All expert gate
and up tensors are Q4_K; the down tensors are split evenly between Q4_K and
Q6_K.

One expert's logical bytes summed over all layers are **137,134,080**. A cold
top-8 decode token would request 1,097,072,640 logical expert bytes if all
selected experts miss in every layer. This is an I/O quantity, not a latency or
cache-performance estimate.

The tensor directory spans offsets 5,933,013 through 5,969,388. GGUF tensor
data starts at offset 5,969,408. The audit read the 7,033-byte HF API response
plus 5,969,389 artifact bytes across 102 contiguous prefix ranges, 5,976,422
bytes total, under the hard 32 MiB limit. Each range response had status 206
and an exact matching `Content-Range`. The last range ended at the directory
boundary; **tensor payload bytes downloaded: 0**.

## Reproduction and checks

Run the pinned remote audit and retain its full tensor directory, including all
absolute offsets, in the ignored build output:

```sh
python -B scripts/inspect-gguf-header.py --json > build/qwen3-target-audit.json
```

The same parser can inspect the local synthetic fixture without network access:

```sh
python -B scripts/inspect-gguf-header.py --local tests/fixtures/moe_paging/qwen3moe-tiny-f32.gguf
```

Offline tests cover GGUF parsing, large string-array skipping, quant block byte
calculation, tensor offsets, unknown type rejection, checked arithmetic, and
mock HTTP 206/`Content-Range`/byte-limit behavior:

```sh
python -B -m unittest tests.test_inspect_gguf_header -v
```

The remote tool defaults to a 32 MiB combined API-and-range budget and a 120 s
overall network deadline. It rejects mismatched revision/file identity,
malformed ranges, truncated or overlapping tensor extents, unsupported GGML
types and any range read that reaches tensor payload bytes.

## Remaining unknowns

- The full payload digest is trusted from HF LFS metadata, not locally verified.
- No tensor payload was read, so quantized values and full-file integrity remain
  unverified.
- No llama.cpp loader or inference run was performed on the target artifact.
- These byte counts do not establish cache residency, PS5 memory fit, storage
  latency, or model quality.
