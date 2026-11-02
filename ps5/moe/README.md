# Bounded GGUF expert source

`ExpertSource` owns read-only POSIX file descriptors for the shard paths and
uses `pread` to copy one expert slice into caller-owned storage. It does not
map shards, allocate a tensor payload, or read beyond the descriptor's expert
stride. The caller owns the fixed staging buffer and must discard it after any
non-`Ok` result.

The caller must validate GGUF tensor metadata before constructing immutable
`ExpertTensorDescriptor` values. Descriptors are checked again for overflow,
shard bounds, unique tensor IDs, consistent shard size/identity, and staging
cap. The source checks each open shard with `fstat`; when device/inode identity
is available it also checks the path still names the opened file on each read.
It checks exact file length on open and before every expert read. Filesystems
without a usable inode identity receive length checks only. Same-size content
corruption cannot be detected without a trusted content hash; this component
does not hash full shards.

Every read returns a structured `ReadError` and is `noexcept`. `pread` retries
`EINTR`, continues after partial reads, reports EOF before the requested slice
as `ShortRead`, and checks cancellation between calls. If it fails after writing
some bytes, the destination is invalid and must not be published as a cache
slot. A future scheduler callback must translate this result into a request
failure, never a resident-weight fallback.

Host command (from the repository root):

```sh
c++ -std=c++17 -O1 -DPS5LM_MOE_EXPERT_SOURCE_TESTING \
    ps5/moe/expert_source_test.cpp ps5/moe/expert_source.cpp \
    -o /tmp/ps5lm_expert_source_test
/tmp/ps5lm_expert_source_test
```

`PS5LM_MOE_EXPERT_SOURCE_TESTING` enables only a deterministic `pread` fault
injection hook used to exercise partial reads, `EINTR`, EOF and cancellation.
Production builds leave the hook out and use POSIX `pread` directly.
