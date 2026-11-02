import contextlib
import importlib.util
import io
from pathlib import Path
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "inspect-gguf-header.py"
FIXTURE = ROOT / "tests" / "fixtures" / "moe_paging" / "qwen3moe-tiny-f32.gguf"
SPEC = importlib.util.spec_from_file_location("inspect_gguf_header", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def pack_string(value):
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def pack_metadata(key, value_type, value):
    return pack_string(key) + struct.pack("<I", value_type) + value


def build_gguf():
    metadata = [
        pack_metadata("general.architecture", 8, pack_string("qwen3moe")),
        pack_metadata("general.alignment", 4, struct.pack("<I", 32)),
        pack_metadata("qwen3moe.block_count", 4, struct.pack("<I", 1)),
        pack_metadata("qwen3moe.embedding_length", 4, struct.pack("<I", 32)),
        pack_metadata("qwen3moe.expert_count", 4, struct.pack("<I", 2)),
        pack_metadata("qwen3moe.expert_used_count", 4, struct.pack("<I", 1)),
        pack_metadata("qwen3moe.expert_feed_forward_length", 4, struct.pack("<I", 32)),
    ]
    tokens = [f"token-{index}" for index in range(2048)]
    token_array = struct.pack("<IQ", 8, len(tokens)) + b"".join(pack_string(token) for token in tokens)
    metadata.append(pack_metadata("tokenizer.ggml.tokens", 9, token_array))

    tensor_specs = [
        ("blk.0.ffn_gate_exps.weight", [32, 32, 2], 2, 1152),
        ("blk.0.ffn_down_exps.weight", [32, 32, 2], 2, 1152),
        ("blk.0.ffn_up_exps.weight", [32, 32, 2], 2, 1152),
        ("blk.0.ffn_gate_inp.weight", [32, 2], 0, 256),
        ("token_embd.weight", [32, 8], 0, 1024),
    ]
    prefix = b"GGUF" + struct.pack("<IQQ", 3, len(tensor_specs), len(metadata))
    prefix += b"".join(metadata)
    tensor_info = bytearray()
    relative_offset = 0
    for name, dims, type_id, byte_count in tensor_specs:
        tensor_info += pack_string(name)
        tensor_info += struct.pack("<I", len(dims))
        tensor_info += struct.pack("<" + "Q" * len(dims), *dims)
        tensor_info += struct.pack("<IQ", type_id, relative_offset)
        relative_offset += ((byte_count + 31) // 32) * 32

    prefix += tensor_info
    data_start = ((len(prefix) + 31) // 32) * 32
    payload = bytearray()
    for _, _, _, byte_count in tensor_specs:
        payload += bytes(byte_count)
        payload += bytes((32 - byte_count % 32) % 32)
    return prefix + bytes(data_start - len(prefix)) + payload, tensor_specs


class MemorySource:
    def __init__(self, data):
        self.data = data
        self.file_size = len(data)

    def read_at(self, offset, length):
        return self.data[offset : offset + length]


class RangeServer:
    def __init__(self, data, response_status=206, bad_content_range=False):
        self.data = data
        self.response_status = response_status
        self.bad_content_range = bad_content_range
        self.ranges = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                range_header = self.headers.get("Range")
                if not range_header or not range_header.startswith("bytes="):
                    self.send_error(400)
                    return
                first, last = (int(part) for part in range_header[6:].split("-", 1))
                owner.ranges.append((first, last))
                body = owner.data[first : last + 1]
                self.send_response(owner.response_status)
                if owner.response_status == 206:
                    content_range = f"bytes {first}-{last}/{len(owner.data)}"
                    if owner.bad_content_range:
                        content_range = f"bytes {first + 1}-{last}/{len(owner.data)}"
                    self.send_header("Content-Range", content_range)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Encoding", "identity")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format_string, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.url = f"http://127.0.0.1:{self.server.server_port}/model.gguf"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class InspectGgufHeaderTests(unittest.TestCase):
    def test_local_qwen3_moe_fixture_and_tensor_accounting(self):
        report = audit.inspect_local(FIXTURE)
        self.assertEqual(report["architecture"]["name"], "qwen3moe")
        self.assertEqual(report["architecture"]["layer_count"], 2)
        self.assertEqual(report["accounting"]["expert_bytes_total"], 49_152)
        self.assertEqual(report["accounting"]["expert_bytes_by_role"], {
            "gate": 16_384,
            "up": 16_384,
            "down": 16_384,
        })
        self.assertEqual(report["accounting"]["non_expert_tensor_bytes"], 101_120)
        self.assertEqual(report["accounting"]["expert_bytes_per_expert_all_layers"], 12_288)
        self.assertEqual(report["transfer"]["tensor_payload_bytes_downloaded"], 0)

    def test_quantized_tensor_sizes_use_block_traits_and_offsets(self):
        data, specs = build_gguf()
        report = audit.parse_gguf(MemorySource(data), {"kind": "memory", "downloaded_bytes": 0})
        accounting = report["accounting"]
        self.assertEqual(accounting["expert_bytes_total"], 3_456)
        self.assertEqual(accounting["expert_bytes_by_role"], {"gate": 1_152, "up": 1_152, "down": 1_152})
        self.assertEqual(accounting["non_expert_tensor_bytes"], 1_280)
        self.assertEqual(accounting["expert_bytes_per_expert_all_layers"], 1_728)

        tensors = {tensor["name"]: tensor for tensor in report["tensors"]}
        for name, _, _, expected_bytes in specs:
            self.assertEqual(tensors[name]["nbytes"], expected_bytes)
            self.assertEqual(tensors[name]["absolute_offset"], report["gguf"]["tensor_data_start_offset"] + tensors[name]["relative_offset"])

    def test_metadata_string_array_is_skipped_without_retaining_elements(self):
        data, _ = build_gguf()
        report = audit.parse_gguf(MemorySource(data), {"kind": "memory", "downloaded_bytes": 0})
        self.assertNotIn("tokenizer.ggml.tokens", report["architecture"])
        self.assertGreater(report["gguf"]["directory_start_offset"], 20_000)

    def test_unknown_quantization_type_is_rejected(self):
        with self.assertRaisesRegex(audit.AuditError, "unknown GGML tensor type 999"):
            audit.tensor_nbytes([32, 32, 2], 999)

    def test_remote_reader_verifies_206_and_bounds_final_range_rounding(self):
        data = FIXTURE.read_bytes()
        budget = audit.DownloadBudget(512 * 1024)
        with RangeServer(data) as server:
            source = audit.HTTPRangeSource(
                server.url,
                len(data),
                budget,
                time.monotonic() + 10,
                quantum=1024,
                max_range_size=4096,
            )
            report = audit.parse_gguf(
                source,
                {"kind": "test-http", "downloaded_bytes": 0},
                range_quantum=1024,
            )
        self.assertGreater(len(server.ranges), 1)
        self.assertEqual(budget.downloaded, source.range_bytes)
        self.assertEqual(source.range_bytes, report["gguf"]["directory_end_offset"])
        self.assertEqual(report["transfer"]["tensor_payload_bytes_downloaded"], 0)
        self.assertEqual(server.ranges[0][0], 0)
        self.assertEqual([list(item) for item in server.ranges], source.ranges)
        for left, right in zip(server.ranges, server.ranges[1:]):
            self.assertEqual(left[1] + 1, right[0])

    def test_remote_reader_rejects_non_partial_or_wrong_content_range(self):
        data, _ = build_gguf()
        for status, bad_range in ((200, False), (206, True)):
            with self.subTest(status=status, bad_range=bad_range):
                with RangeServer(data, response_status=status, bad_content_range=bad_range) as server:
                    source = audit.HTTPRangeSource(
                        server.url,
                        len(data),
                        audit.DownloadBudget(4096),
                        time.monotonic() + 10,
                        quantum=1024,
                        max_range_size=4096,
                    )
                    with self.assertRaises(audit.AuditError):
                        source.read_at(0, 4)

    def test_remote_reader_never_exceeds_hard_byte_budget(self):
        data, _ = build_gguf()
        budget_limit = 128
        budget = audit.DownloadBudget(budget_limit)
        with RangeServer(data) as server:
            source = audit.HTTPRangeSource(
                server.url,
                len(data),
                budget,
                time.monotonic() + 10,
                quantum=1024,
                max_range_size=4096,
            )
            with self.assertRaisesRegex(audit.AuditError, "download limit"):
                audit.parse_gguf(source, {"kind": "test-http", "downloaded_bytes": 0})
        self.assertLessEqual(budget.downloaded, budget_limit)

    def test_gguf_tensor_sizes_reject_overflow(self):
        with self.assertRaisesRegex(audit.AuditError, "overflows"):
            audit.checked_mul(audit.U64_MAX, 2, "test product")

    def test_cli_enforces_hard_download_and_deadline_caps_before_network(self):
        for args, message in (
            (["--max-download-mib", "33"], "cannot exceed 32 MiB"),
            (["--timeout", "301"], "cannot exceed 300 seconds"),
        ):
            with self.subTest(args=args):
                with contextlib.redirect_stderr(io.StringIO()) as error_output:
                    result = audit.main(args)
                self.assertEqual(result, 2)
                self.assertIn(message, error_output.getvalue())


if __name__ == "__main__":
    unittest.main()
