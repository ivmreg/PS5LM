import contextlib
import importlib.util
import io
import json
import os
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock


sys.dont_write_bytecode = True
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "benchmark-ps5lm.py")
SPEC = importlib.util.spec_from_file_location("benchmark_ps5lm", SCRIPT_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def _data(payload, line_ending="\n"):
    return "data: {}\n\n".format(json.dumps(payload, ensure_ascii=False)).replace(
        "\n", line_ending
    ).encode("utf-8")


def _successful_stream(include_usage=True):
    events = [
        {
            "id": "completion-test",
            "model": "mock-llama-model",
            "choices": [{"delta": {"role": "assistant"}, "finish_reason": None}],
        },
        {
            "choices": [
                {"delta": {"reasoning_content": "private reasoning"}, "finish_reason": None}
            ]
        },
        {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": " 🌍"}, "finish_reason": None}]},
        {
            "choices": [
                {"delta": {}, "finish_reason": "stop"}
            ]
        },
    ]
    body = b"".join(_data(event, line_ending="\r\n") for event in events)
    if include_usage:
        body += _data(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 4,
                    "total_tokens": 13,
                },
                "timings": {
                    "prompt_n": 17,
                    "prompt_ms": 2.4,
                    "predicted_n": 4,
                    "predicted_ms": 8.1,
                    "predicted_per_second": 493.8,
                },
            }
        )
    body += b"data: [DONE]\r\n\r\n"
    return body


class MockEndpoint:
    def __init__(self, responder):
        self.requests = []
        self.authorization_headers = []
        self.responder = responder
        self.httpd = None
        self.thread = None

        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                payload = json.loads(body.decode("utf-8"))
                owner.requests.append(payload)
                owner.authorization_headers.append(self.headers.get("Authorization"))
                response = owner.responder(payload, len(owner.requests))
                response_body = response.get("body", b"")
                if isinstance(response_body, str):
                    response_body = response_body.encode("utf-8")
                scheduled_chunks = response.get("chunks")
                if scheduled_chunks is not None:
                    response_body = b"".join(chunk for delay, chunk in scheduled_chunks)

                raw_header_chunks = response.get("raw_header_chunks")
                if raw_header_chunks is not None:
                    try:
                        for delay, chunk in raw_header_chunks:
                            if delay:
                                time.sleep(delay)
                            self.connection.sendall(chunk)
                        self.connection.sendall(response_body)
                    except OSError:
                        pass
                    self.close_connection = True
                    return

                self.send_response(response.get("status", 200))
                headers = dict(response.get("headers", {}))
                headers.setdefault("Content-Type", "text/event-stream; charset=utf-8")
                headers.setdefault("Connection", "close")
                headers.setdefault("Content-Length", str(len(response_body)))
                for name, value in headers.items():
                    self.send_header(name, value)
                self.end_headers()
                self.close_connection = True

                if scheduled_chunks is not None:
                    fragments = scheduled_chunks
                else:
                    fragment_size = response.get("fragment_size", len(response_body) or 1)
                    delay = response.get("delay_between_fragments", 0)
                    fragments = [
                        (delay if offset else 0, response_body[offset : offset + fragment_size])
                        for offset in range(0, len(response_body), fragment_size)
                    ]
                for delay, fragment in fragments:
                    if delay:
                        time.sleep(delay)
                    try:
                        self.wfile.write(fragment)
                        self.wfile.flush()
                    except OSError:
                        break

            def log_message(self, format_string, *args):
                pass

        self.handler_type = Handler

    def __enter__(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), self.handler_type)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:{}/v1/chat/completions".format(self.httpd.server_port)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)


def _response(body=None, **kwargs):
    result = {"status": 200, "body": _successful_stream(), "fragment_size": 4096}
    if body is not None:
        result["body"] = body
    result.update(kwargs)
    return result


class BenchmarkClientTests(unittest.TestCase):
    def _trial(self, server, timeout=2.0):
        return benchmark.run_trial(
            url=server.url,
            prompt="test prompt",
            prompt_id="test",
            kind="repeat",
            index=1,
            requested_model="mock-request-name",
            timeout_seconds=timeout,
            api_key=None,
        )

    def test_fragmented_sse_done_and_client_server_measurements(self):
        body = _successful_stream(include_usage=True)
        with MockEndpoint(
            lambda payload, index: _response(
                body=body,
                fragment_size=3,
                headers={
                    "Server-Timing": "total;dur=11.2",
                    "x-request-id": "request-123",
                },
            )
        ) as server:
            trial = self._trial(server)

        self.assertEqual(trial["status"], "ok")
        self.assertTrue(trial["done_received"])
        self.assertEqual(trial["output_text"], "Hello 🌍")
        self.assertIsNotNone(trial["first_role_ms"])
        self.assertIsNotNone(trial["first_reasoning_ms"])
        self.assertIsNotNone(trial["first_content_ms"])
        self.assertEqual(trial["model_identity"], "mock-llama-model")
        self.assertEqual(trial["response_id"], "completion-test")
        self.assertEqual(trial["server_reported"]["usage"]["total_tokens"], 13)
        self.assertEqual(
            trial["server_reported"]["timings"][0]["predicted_per_second"], 493.8
        )
        self.assertEqual(trial["server_identity"]["server_timing_header"], "total;dur=11.2")
        self.assertEqual(trial["server_identity"]["request_id"], "request-123")
        self.assertEqual(server.requests[0]["cache_prompt"], False)
        self.assertEqual(server.requests[0]["temperature"], 0)
        self.assertEqual(server.requests[0]["seed"], 1)
        self.assertEqual(server.requests[0]["stream_options"], {"include_usage": True})
        self.assertEqual(server.requests[0]["max_tokens"], 256)

    def test_missing_usage_is_success_and_remains_unavailable(self):
        with MockEndpoint(lambda payload, index: _response(body=_successful_stream(False))) as server:
            trial = self._trial(server)
        self.assertEqual(trial["status"], "ok")
        self.assertIsNone(trial["server_reported"]["usage"])
        self.assertIsNone(trial["server_reported"]["timings"])

    def test_sse_error_preserves_partial_trial(self):
        body = _data({"choices": [{"delta": {"content": "partial"}, "finish_reason": None}]})
        body += b"event: error\ndata: {\"error\":{\"message\":\"synthetic failure\"}}\n\n"
        with MockEndpoint(lambda payload, index: _response(body=body)) as server:
            trial = self._trial(server)
        self.assertEqual(trial["status"], "failed")
        self.assertEqual(trial["output_text"], "partial")
        self.assertEqual(trial["error"]["type"], "server_error")
        self.assertIn("synthetic failure", trial["error"]["message"])

    def test_http_error_is_retained(self):
        with MockEndpoint(
            lambda payload, index: {
                "status": 503,
                "headers": {"Content-Type": "application/json"},
                "body": b'{"error":{"message":"service unavailable"}}',
            }
        ) as server:
            trial = self._trial(server)
        self.assertEqual(trial["status"], "failed")
        self.assertEqual(trial["http_status"], 503)
        self.assertEqual(trial["error"]["type"], "http_error")
        self.assertIn("service unavailable", trial["error"]["message"])

    def test_eof_without_done_is_stream_truncation(self):
        body = _data({"choices": [{"delta": {"content": "partial"}, "finish_reason": "stop"}]})
        with MockEndpoint(lambda payload, index: _response(body=body)) as server:
            trial = self._trial(server)
        self.assertEqual(trial["status"], "failed")
        self.assertEqual(trial["error"]["type"], "stream_truncated")
        self.assertEqual(trial["output_text"], "partial")

    def test_done_without_completion_chunk_is_not_a_success(self):
        with MockEndpoint(lambda payload, index: _response(body=b"data: [DONE]\n\n")) as server:
            trial = self._trial(server)
        self.assertEqual(trial["status"], "failed")
        self.assertEqual(trial["error"]["type"], "empty_stream")

    def test_idle_socket_timeout_is_a_failed_trial(self):
        body = _successful_stream()
        with MockEndpoint(
            lambda payload, index: _response(
                body=body, fragment_size=8, delay_between_fragments=0.25
            )
        ) as server:
            trial = self._trial(server, timeout=0.06)
        self.assertEqual(trial["status"], "failed")
        self.assertEqual(trial["error"]["type"], "timeout")

    def test_overall_deadline_bounds_trickling_stream(self):
        body = _successful_stream()
        with MockEndpoint(
            lambda payload, index: _response(
                body=body, fragment_size=16, delay_between_fragments=0.025
            )
        ) as server:
            trial = self._trial(server, timeout=0.16)
        self.assertEqual(trial["status"], "failed")
        self.assertEqual(trial["error"]["type"], "timeout")

    def test_close_connection_trickles_then_stalls_at_overall_deadline(self):
        role_event = _data(
            {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]}
        )
        content_event = _data(
            {"choices": [{"delta": {"content": "partial"}, "finish_reason": None}]}
        )
        scheduled_chunks = [
            (0, role_event),
            (0.055, content_event),
            (0.055, b"data: ["),
            (0.4, b"DONE]\n\n"),
        ]
        with MockEndpoint(
            lambda payload, index: {
                "status": 200,
                "headers": {"Connection": "close"},
                "chunks": scheduled_chunks,
            }
        ) as server:
            trial = self._trial(server, timeout=0.18)
        self.assertEqual(trial["status"], "failed")
        self.assertEqual(trial["error"]["type"], "timeout")
        self.assertEqual(trial["output_text"], "partial")

    def test_trickling_response_headers_are_bounded_by_overall_deadline(self):
        raw_headers = [
            (0, b"HTTP/1.1 200 OK\r\n"),
            (0.05, b"Content-Type: text/event-stream\r\n"),
            (0.05, b"Connection: close\r\n"),
            (0.05, b"Content-Length: 32\r\n"),
            (0.05, b"\r\n"),
        ]
        with MockEndpoint(
            lambda payload, index: {
                "status": 200,
                "body": _successful_stream(),
                "raw_header_chunks": raw_headers,
            }
        ) as server:
            trial = self._trial(server, timeout=0.18)
        self.assertEqual(trial["status"], "failed")
        self.assertEqual(trial["error"]["type"], "timeout")


    def test_warmup_is_separate_and_cli_retains_failed_trials(self):
        def respond(payload, request_number):
            if request_number == 1:
                body = b'event: error\ndata: {"error":{"message":"warmup failed"}}\n\n'
                return _response(body=body)
            return _response()

        output_path = os.path.join(REPO_ROOT, "build", "benchmark-test-output.json")
        try:
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "dummy-test-secret"}):
                with MockEndpoint(respond) as server:
                    with contextlib.redirect_stdout(io.StringIO()):
                        exit_code = benchmark.main(
                            [
                                "--url",
                                server.url,
                                "--warmup",
                                "--repeats",
                                "1",
                                "--max-tokens",
                                "192",
                                "--timeout",
                                "2",
                                "--output",
                                output_path,
                            ]
                        )
            with open(output_path, "r", encoding="utf-8") as source:
                report = json.load(source)
            self.assertTrue(all(header is None for header in server.authorization_headers))
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

        self.assertEqual(exit_code, 1)
        self.assertTrue(report["run"]["warmup_included"])
        self.assertEqual(report["trials"][0]["kind"], "warmup")
        self.assertEqual(report["trials"][0]["status"], "failed")
        self.assertEqual(report["trials"][1]["kind"], "repeat")
        self.assertEqual(
            [trial["kind"] for trial in report["trials"][2:]],
            ["unrelated", "unrelated", "unrelated"],
        )
        self.assertEqual(report["summary"]["trial_count"], 5)
        self.assertEqual(report["summary"]["failed_trial_count"], 1)
        self.assertEqual(report["run"]["max_tokens"], 192)
        self.assertEqual(report["run"]["request_parameters"]["max_tokens"], 192)
        self.assertIsNone(report["run"]["api_key_environment_variable"])
        self.assertNotIn("dummy-test-secret", json.dumps(report))

    def test_cli_help(self):
        with self.assertRaises(SystemExit) as raised:
            benchmark.main(["--help"])
        self.assertEqual(raised.exception.code, 0)

    def test_cli_rejects_nonfinite_timeout_before_request(self):
        for value in ("nan", "inf", "-inf"):
            with self.subTest(timeout=value):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        benchmark.main(
                            [
                                "--url",
                                "http://127.0.0.1:1/v1/chat/completions",
                                "--timeout",
                                value,
                            ]
                        )
                self.assertEqual(raised.exception.code, 2)

    def test_server_reported_draft_metrics_recorded(self):
        events = [
            {"choices": [{"delta": {"content": "ok"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
                "timings": {
                    "prompt_n": 5,
                    "predicted_n": 10,
                    "draft_n": 10,
                    "draft_n_accepted": 8,
                },
            },
        ]
        body = b"".join(_data(event, line_ending="\r\n") for event in events) + b"data: [DONE]\r\n\r\n"
        with MockEndpoint(
            lambda payload, index: _response(body=body, fragment_size=4)
        ) as server:
            trial = self._trial(server)

        self.assertEqual(trial["status"], "ok")
        self.assertEqual(trial["server_reported"]["draft_tokens"], 10)
        self.assertEqual(trial["server_reported"]["draft_accepted"], 8)
        self.assertEqual(trial["server_reported"]["draft_acceptance_rate"], 0.8)

    def test_prompt_set_selection(self):
        for pset, expected_id in (("general", "ps5lm-streaming-v1"), ("code", "ps5lm-code-v1"), ("prose", "ps5lm-prose-v1")):
            schedule = benchmark.build_schedule(repeats=2, include_warmup=True, prompt_set=pset)
            self.assertEqual(schedule[0][0], "warmup")
            self.assertEqual(schedule[1][0], "repeat")
            self.assertEqual(schedule[2][0], "repeat")
            self.assertEqual(len(schedule), 6)  # 1 warmup + 2 repeats + 3 unrelated
            config = benchmark.PROMPT_SETS[pset]
            self.assertEqual(config["id"], expected_id)


if __name__ == "__main__":
    unittest.main()
