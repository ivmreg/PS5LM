import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "moe_paging"
FIXTURE_PATH = FIXTURE_DIR / "qwen3moe-tiny-f32.gguf"
GENERATOR = ROOT / "scripts" / "create-test-moe.py"

sys.path.insert(0, str(ROOT / "third_party" / "llama.cpp" / "gguf-py"))
import gguf


def expected_tensor_shapes():
    embedding = 32
    vocabulary = 260
    head_dim = 8
    experts = 4
    expert_width = 16

    shapes = {
        "token_embd.weight": [embedding, vocabulary],
        "output_norm.weight": [embedding],
        "output.weight": [embedding, vocabulary],
    }
    for layer in range(2):
        prefix = f"blk.{layer}."
        shapes.update(
            {
                prefix + "attn_norm.weight": [embedding],
                prefix + "attn_q.weight": [embedding, embedding],
                prefix + "attn_k.weight": [embedding, embedding],
                prefix + "attn_v.weight": [embedding, embedding],
                prefix + "attn_output.weight": [embedding, embedding],
                prefix + "attn_k_norm.weight": [head_dim],
                prefix + "attn_q_norm.weight": [head_dim],
                prefix + "ffn_norm.weight": [embedding],
                prefix + "ffn_gate_inp.weight": [embedding, experts],
                prefix + "ffn_gate_exps.weight": [embedding, expert_width, experts],
                prefix + "ffn_down_exps.weight": [expert_width, embedding, experts],
                prefix + "ffn_up_exps.weight": [embedding, expert_width, experts],
            }
        )
    return shapes


def find_loader_binary():
    configured = os.environ.get("LLAMA_MOE_TEST_BINARY") or os.environ.get("LLAMA_CLI")
    if configured:
        return configured

    candidates = [
        ROOT / "build" / "moe-host" / "bin" / "llama-completion",
        ROOT / "build" / "moe-host" / "bin" / "llama-completion.exe",
        ROOT / "third_party" / "llama.cpp" / "build" / "bin" / "llama-cli.exe",
        ROOT / "third_party" / "llama.cpp" / "build" / "bin" / "llama-cli",
        ROOT / "third_party" / "llama.cpp" / "build" / "bin" / "llama-completion.exe",
        ROOT / "third_party" / "llama.cpp" / "build" / "bin" / "llama-completion",
        ROOT / "third_party" / "llama.cpp" / "build" / "Release" / "llama-cli.exe",
        ROOT / "third_party" / "llama.cpp" / "build" / "llama-cli.exe",
        ROOT / "build" / "llama-cli.exe",
        ROOT / "build" / "llama-cli",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return shutil.which("llama-cli") or shutil.which("llama-completion")


class MoeFixtureTests(unittest.TestCase):
    def test_gguf_metadata_and_full_tensor_schema(self):
        reader = gguf.GGUFReader(FIXTURE_PATH)
        fields = reader.fields
        self.assertEqual(fields["general.architecture"].contents(), "qwen3moe")
        self.assertEqual(fields["qwen3moe.vocab_size"].contents(), 260)
        self.assertEqual(fields["qwen3moe.embedding_length"].contents(), 32)
        self.assertEqual(fields["qwen3moe.block_count"].contents(), 2)
        self.assertEqual(fields["qwen3moe.expert_count"].contents(), 4)
        self.assertEqual(fields["qwen3moe.expert_used_count"].contents(), 2)
        self.assertEqual(fields["qwen3moe.expert_feed_forward_length"].contents(), 16)
        self.assertEqual(fields["tokenizer.ggml.model"].contents(), "gpt2")
        self.assertEqual(fields["tokenizer.ggml.pre"].contents(), "qwen2")

        tokens = fields["tokenizer.ggml.tokens"].contents()
        merges = fields["tokenizer.ggml.merges"].contents()
        token_types = fields["tokenizer.ggml.token_type"].contents()
        self.assertEqual(len(tokens), 260)
        self.assertEqual(merges, [f"{tokens[0]} {tokens[1]}"])
        self.assertEqual(tokens[-4:], ["<|endoftext|>", "<|im_start|>", "<|im_sep|>", "<|im_end|>"])
        self.assertEqual(token_types[:256], [gguf.TokenType.NORMAL] * 256)
        self.assertEqual(token_types[256:], [gguf.TokenType.CONTROL] * 4)
        self.assertEqual(fields["tokenizer.ggml.bos_token_id"].contents(), 256)
        self.assertEqual(fields["tokenizer.ggml.eos_token_id"].contents(), 259)

        expected_shapes = expected_tensor_shapes()
        actual_tensors = {tensor.name: tensor for tensor in reader.tensors}
        self.assertEqual(set(actual_tensors), set(expected_shapes))
        for name, shape in expected_shapes.items():
            with self.subTest(tensor=name):
                tensor = actual_tensors[name]
                self.assertEqual(tensor.shape.tolist(), shape)
                self.assertEqual(tensor.tensor_type, gguf.GGMLQuantizationType.F32)

    def test_generator_reproduces_fixture_bytes(self):
        temporary = tempfile.NamedTemporaryFile(
            prefix=".moe-fixture-repro-", suffix=".gguf", dir=FIXTURE_DIR, delete=False
        )
        temporary_path = Path(temporary.name)
        temporary.close()
        try:
            result = subprocess.run(
                [sys.executable, str(GENERATOR), "--output", str(temporary_path)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(temporary_path.read_bytes(), FIXTURE_PATH.read_bytes())
        finally:
            temporary_path.unlink(missing_ok=True)

    def test_cpu_loader_and_one_token_inference_when_binary_is_available(self):
        binary = find_loader_binary()
        if binary is None:
            self.skipTest("set LLAMA_MOE_TEST_BINARY to a llama-cli or llama-completion binary")

        result = subprocess.run(
            [
                binary,
                "-m",
                str(FIXTURE_PATH),
                "-c",
                "32",
                "-p",
                "a",
                "-n",
                "1",
                "-ngl",
                "0",
                "--temp",
                "0",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
