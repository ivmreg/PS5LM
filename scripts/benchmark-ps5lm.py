#!/usr/bin/env python3
"""Benchmark an already-running OpenAI-compatible streaming chat endpoint.

Only client-visible timings are measured here. Server timing and token fields
are copied through when the server supplies them; the client never estimates
token counts, decode rates, memory use, storage traffic, or model load time.
"""

import argparse
import datetime
import http.client
import json
import math
import os
import platform
import socket
import ssl
import sys
import threading
import time
from urllib.parse import urlsplit


CLIENT_VERSION = "1"
PROMPT_SET = "ps5lm-streaming-v1"

REPEAT_PROMPT = (
    "In about 120 words, explain how an HTTP request reaches a server, how a "
    "streaming response is delivered, and why time-to-first-content differs "
    "from total response time. Include one concrete example and avoid bullets."
)
WARMUP_PROMPT = "Reply with the single word ready."
UNRELATED_PROMPTS = (
    (
        "compass",
        "In about 100 words, explain how a magnetic compass supports navigation, "
        "what makes its needle move, and one limitation of using it. Avoid bullets.",
    ),
    (
        "leaf-color",
        "In about 100 words, explain why many leaves change color in autumn, "
        "which pigments are involved, and how weather can affect the display. Avoid bullets.",
    ),
    (
        "checksum",
        "In about 100 words, explain how a checksum can help detect a damaged "
        "file, what it cannot prove, and how a sender and receiver use it. Avoid bullets.",
    ),
)

PROMPT_SETS = {
    "general": {
        "id": "ps5lm-streaming-v1",
        "repeat": REPEAT_PROMPT,
        "warmup": WARMUP_PROMPT,
        "unrelated": UNRELATED_PROMPTS,
    },
    "code": {
        "id": "ps5lm-code-v1",
        "repeat": (
            "Write a Python function `quicksort(arr)` that sorts a list of integers in ascending "
            "order. Include a concise docstring and comments explaining the partition step and base case."
        ),
        "warmup": "def ready(): return True",
        "unrelated": (
            (
                "binary-search",
                "Write a Python function `binary_search(arr, target)` that returns the index of target "
                "in a sorted list or -1 if not found. Include docstring and comments.",
            ),
            (
                "lru-cache",
                "Write a minimal Python class `LRUCache(capacity)` with `get(key)` and `put(key, value)` "
                "methods using standard library collections. Include comments.",
            ),
            (
                "merge-sorted",
                "Write a Python function `merge_sorted(a, b)` that merges two sorted lists into one sorted list "
                "in linear time. Include docstring.",
            ),
        ),
    },
    "prose": {
        "id": "ps5lm-prose-v1",
        "repeat": (
            "In about 120 words, describe the atmosphere of a quiet coastal library in late autumn as rain "
            "taps against the tall windows. Use vivid sensory details and avoid bullets."
        ),
        "warmup": "Ready.",
        "unrelated": (
            (
                "mountain-dawn",
                "In about 100 words, portray the sunrise over a snowy mountain range, highlighting changes in "
                "light, shadows, and crisp morning air. Avoid bullets.",
            ),
            (
                "desert-night",
                "In about 100 words, depict the stillness of a desert at midnight under a clear sky filled with "
                "stars. Avoid bullets.",
            ),
            (
                "harbor-dusk",
                "In about 100 words, describe the scents and sounds of an old fishing harbor at twilight as the "
                "boats return to their docks. Avoid bullets.",
            ),
        ),
    },
}

BASE_REQUEST_PARAMETERS = {
    "stream": True,
    "stream_options": {"include_usage": True},
    "temperature": 0,
    "top_p": 1,
    "seed": 1,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    # llama.cpp extension: avoid reusing server-side prompt cache between runs.
    "cache_prompt": False,
}


def request_parameters(max_tokens):
    parameters = dict(BASE_REQUEST_PARAMETERS)
    parameters["max_tokens"] = max_tokens
    return parameters


class BenchmarkError(Exception):
    """A request or stream failed in a way that should be retained in output."""

    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind
        self.message = message


class SSEParser:
    """Small incremental SSE parser that tolerates arbitrary byte boundaries."""

    MAX_EVENT_BYTES = 1024 * 1024

    def __init__(self):
        self._line_buffer = bytearray()
        self._data_lines = []
        self._event_name = "message"
        self._event_size = 0
        self._first_line = True

    def feed(self, chunk):
        self._line_buffer.extend(chunk)
        if len(self._line_buffer) > self.MAX_EVENT_BYTES:
            raise BenchmarkError("event_too_large", "SSE line exceeded 1 MiB")
        events = []
        while True:
            newline = self._line_buffer.find(b"\n")
            if newline < 0:
                break
            line = bytes(self._line_buffer[:newline])
            del self._line_buffer[: newline + 1]
            if line.endswith(b"\r"):
                line = line[:-1]
            event = self._consume_line(line)
            if event is not None:
                events.append(event)
        return events

    def finish(self):
        events = []
        if self._line_buffer:
            line = bytes(self._line_buffer)
            self._line_buffer.clear()
            if line.endswith(b"\r"):
                line = line[:-1]
            event = self._consume_line(line)
            if event is not None:
                events.append(event)
        event = self._dispatch()
        if event is not None:
            events.append(event)
        return events

    def _consume_line(self, raw_line):
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BenchmarkError("invalid_sse", "SSE stream contained invalid UTF-8") from exc

        if self._first_line:
            self._first_line = False
            if line.startswith("\ufeff"):
                line = line[1:]
        if line == "":
            return self._dispatch()
        if line.startswith(":"):
            return None

        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "data":
            self._event_size += len(value.encode("utf-8")) + 1
            if self._event_size > self.MAX_EVENT_BYTES:
                raise BenchmarkError("event_too_large", "SSE event exceeded 1 MiB")
            self._data_lines.append(value)
        elif field == "event":
            self._event_name = value
        return None

    def _dispatch(self):
        if not self._data_lines:
            self._event_name = "message"
            self._event_size = 0
            return None
        event = (self._event_name, "\n".join(self._data_lines))
        self._data_lines = []
        self._event_name = "message"
        self._event_size = 0
        return event


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_endpoint(url):
    """Keep endpoint identity while avoiding query-string credential disclosure."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if ":" in host:
        host = "[{}]".format(host)
    netloc = host
    if parts.port is not None:
        netloc += ":{}".format(parts.port)
    query = "?<redacted>" if parts.query else ""
    return "{}://{}{}{}".format(parts.scheme, netloc, parts.path, query)


def _parse_endpoint(url):
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("--url must be an absolute http:// or https:// endpoint URL")
    if parts.username is not None or parts.password is not None:
        raise ValueError("credentials in --url are not supported; use the API key environment variable")
    if parts.fragment:
        raise ValueError("--url must not contain a fragment")
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError("--url contains an invalid port") from exc
    target = parts.path or "/"
    if parts.query:
        target += "?" + parts.query
    return parts, target, port


def _set_remaining_timeout(connection, deadline, transport_socket=None):
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise BenchmarkError("timeout", "overall trial deadline exceeded")
    target_socket = transport_socket if transport_socket is not None else connection.sock
    if target_socket is not None:
        target_socket.settimeout(remaining)


def _raise_if_deadline_expired(deadline, expired_event):
    if expired_event.is_set() or time.perf_counter() >= deadline:
        raise BenchmarkError("timeout", "overall trial deadline exceeded")


def _read_response_chunk(response, connection, deadline, transport_socket, size=8192):
    _set_remaining_timeout(connection, deadline, transport_socket)
    # read1 returns currently available bytes instead of waiting to fill the buffer.
    return response.read1(size)


def _expire_socket(sock, expired_event):
    expired_event.set()
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass


def _nonempty(value):
    if isinstance(value, str):
        return value != ""
    return value is not None and bool(value)


def _content_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if not isinstance(item, dict):
                continue
            if item.get("type") in ("text", "output_text"):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _format_server_error(error):
    if isinstance(error, dict):
        message = error.get("message") or error.get("type") or json.dumps(error, ensure_ascii=False)
        return {"type": "server_error", "message": str(message), "details": error}
    return {"type": "server_error", "message": str(error), "details": error}


def _apply_sse_event(event_name, data, trial, started):
    if data.strip() == "[DONE]":
        trial["done_received"] = True
        return True

    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise BenchmarkError("invalid_sse_json", "SSE data was not valid JSON: {}".format(exc)) from exc
    if not isinstance(payload, dict):
        raise BenchmarkError("invalid_sse_json", "SSE JSON event was not an object")
    if event_name == "error" or payload.get("error") is not None:
        error = payload.get("error", payload)
        details = _format_server_error(error)
        raise BenchmarkError(details["type"], details["message"])

    if payload.get("id") is not None and trial["response_id"] is None:
        trial["response_id"] = payload.get("id")
    model = payload.get("model")
    if isinstance(model, str) and model and model not in trial["model_identities"]:
        trial["model_identities"].append(model)
        if trial["model_identity"] is None:
            trial["model_identity"] = model
    fingerprint = payload.get("system_fingerprint")
    if isinstance(fingerprint, str) and fingerprint and fingerprint not in trial["system_fingerprints"]:
        trial["system_fingerprints"].append(fingerprint)

    usage = payload.get("usage")
    if isinstance(usage, dict):
        # Keep server-provided counts and details exactly as sent; do not estimate them.
        trial["server_reported"]["usage"] = usage
    timings = payload.get("timings")
    if timings is not None:
        if trial["server_reported"]["timings"] is None:
            trial["server_reported"]["timings"] = []
        trial["server_reported"]["timings"].append(timings)
        if isinstance(timings, dict):
            draft_n = timings.get("draft_n")
            draft_acc = timings.get("draft_n_accepted")
            if draft_n is not None:
                trial["server_reported"]["draft_tokens"] = draft_n
            if draft_acc is not None:
                trial["server_reported"]["draft_accepted"] = draft_acc
            if draft_n is not None and draft_acc is not None and draft_n > 0:
                trial["server_reported"]["draft_acceptance_rate"] = round(draft_acc / draft_n, 4)

    choices = payload.get("choices", [])
    if choices is None:
        choices = []
    if not isinstance(choices, list):
        raise BenchmarkError("invalid_sse_json", "choices field was not a list")
    if choices:
        trial["completion_chunk_count"] += 1
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and finish_reason not in trial["finish_reasons"]:
            trial["finish_reasons"].append(finish_reason)
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue

        role = delta.get("role")
        if isinstance(role, str) and role and trial["first_role_ms"] is None:
            trial["first_role_ms"] = round((time.perf_counter() - started) * 1000, 3)

        has_reasoning = any(
            _nonempty(value)
            for key, value in delta.items()
            if "reasoning" in key.lower() or key.lower() == "analysis"
        )
        if has_reasoning and trial["first_reasoning_ms"] is None:
            trial["first_reasoning_ms"] = round((time.perf_counter() - started) * 1000, 3)

        content = _content_text(delta.get("content"))
        if content:
            if trial["first_content_ms"] is None:
                trial["first_content_ms"] = round((time.perf_counter() - started) * 1000, 3)
            trial["output_text"] += content
    return False


def _base_trial(index, kind, prompt_id, prompt, requested_model):
    return {
        "index": index,
        "kind": kind,
        "prompt_id": prompt_id,
        "prompt": prompt,
        "requested_model": requested_model,
        "status": "failed",
        "started_at_utc": None,
        "http_status": None,
        "wall_time_ms": None,
        "first_role_ms": None,
        "first_reasoning_ms": None,
        "first_content_ms": None,
        "output_text": "",
        "finish_reasons": [],
        "model_identity": None,
        "model_identities": [],
        "system_fingerprints": [],
        "response_id": None,
        "server_identity": {
            "server_header": None,
            "server_date": None,
            "request_id": None,
            "server_timing_header": None,
        },
        "server_reported": {
            "usage": None,
            "timings": None,
            "draft_tokens": None,
            "draft_accepted": None,
            "draft_acceptance_rate": None,
        },
        "completion_chunk_count": 0,
        "done_received": False,
        "error": None,
    }


def run_trial(url, prompt, prompt_id, kind, index, requested_model, timeout_seconds, api_key,
              max_tokens=256):
    trial = _base_trial(index, kind, prompt_id, prompt, requested_model)
    started = time.perf_counter()
    deadline = started + timeout_seconds
    trial["started_at_utc"] = _utc_now()
    connection = None
    response = None
    transport_socket = None
    deadline_timer = None
    deadline_expired = threading.Event()
    try:
        parts, target, port = _parse_endpoint(url)
        connect_timeout = deadline - time.perf_counter()
        if connect_timeout <= 0:
            raise BenchmarkError("timeout", "overall trial deadline exceeded before connecting")
        if parts.scheme == "https":
            connection = http.client.HTTPSConnection(
                parts.hostname,
                port=port,
                timeout=connect_timeout,
                context=ssl.create_default_context(),
            )
        else:
            connection = http.client.HTTPConnection(parts.hostname, port=port, timeout=connect_timeout)

        request = {
            "model": requested_model,
            "messages": [{"role": "user", "content": prompt}],
        }
        request.update(request_parameters(max_tokens))
        body = json.dumps(request, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

        connection.connect()
        transport_socket = connection.sock
        _set_remaining_timeout(connection, deadline, transport_socket)
        remaining = deadline - time.perf_counter()
        deadline_timer = threading.Timer(
            remaining, _expire_socket, args=(transport_socket, deadline_expired)
        )
        deadline_timer.daemon = True
        deadline_timer.start()
        connection.putrequest("POST", target)
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Accept", "text/event-stream")
        connection.putheader("Connection", "close")
        connection.putheader("Content-Length", str(len(body)))
        if api_key:
            connection.putheader("Authorization", "Bearer {}".format(api_key))
        connection.endheaders(body)
        _set_remaining_timeout(connection, deadline, transport_socket)
        # HTTPConnection.getresponse() closes its socket when the response
        # declares Connection: close. Keep the socket wrapper alive while the
        # response's makefile is being read so its timeout can track deadline.
        close_connection = connection.close
        connection.close = lambda: None
        try:
            response = connection.getresponse()
        finally:
            connection.close = close_connection
        _raise_if_deadline_expired(deadline, deadline_expired)
        trial["http_status"] = response.status
        trial["server_identity"].update(
            {
                "server_header": response.getheader("Server"),
                "server_date": response.getheader("Date"),
                "request_id": response.getheader("x-request-id")
                or response.getheader("x-correlation-id"),
                "server_timing_header": response.getheader("Server-Timing"),
            }
        )

        if response.status < 200 or response.status >= 300:
            error_bytes = bytearray()
            while len(error_bytes) < 8192:
                chunk = _read_response_chunk(
                    response,
                    connection,
                    deadline,
                    transport_socket,
                    min(4096, 8192 - len(error_bytes)),
                )
                _raise_if_deadline_expired(deadline, deadline_expired)
                if not chunk:
                    break
                error_bytes.extend(chunk)
            error_text = bytes(error_bytes).decode("utf-8", errors="replace")
            try:
                error_body = json.loads(error_text) if error_text else None
            except json.JSONDecodeError:
                error_body = error_text
            if isinstance(error_body, dict) and error_body.get("error") is not None:
                server_error = _format_server_error(error_body["error"])
                message = server_error["message"]
            else:
                message = error_text.strip() or "HTTP {}".format(response.status)
            raise BenchmarkError("http_error", "HTTP {}: {}".format(response.status, message))

        parser = SSEParser()
        while True:
            chunk = _read_response_chunk(response, connection, deadline, transport_socket)
            _raise_if_deadline_expired(deadline, deadline_expired)
            if not chunk:
                for event_name, data in parser.finish():
                    if _apply_sse_event(event_name, data, trial, started):
                        break
                break
            stop = False
            for event_name, data in parser.feed(chunk):
                if _apply_sse_event(event_name, data, trial, started):
                    stop = True
                    break
            if stop:
                break

        if not trial["done_received"]:
            raise BenchmarkError("stream_truncated", "SSE stream ended before the [DONE] marker")
        if trial["completion_chunk_count"] == 0:
            raise BenchmarkError(
                "empty_stream", "[DONE] arrived without a completion chunk"
            )
        trial["status"] = "ok"
    except BenchmarkError as exc:
        trial["error"] = {"type": exc.kind, "message": exc.message}
    except (socket.timeout, TimeoutError) as exc:
        trial["error"] = {
            "type": "timeout",
            "message": "request exceeded the {:.3g}s overall timeout ({})".format(
                timeout_seconds, str(exc) or "socket read timed out"
            ),
        }
    except (OSError, http.client.HTTPException, ssl.SSLError, ValueError) as exc:
        if deadline_expired.is_set() or time.perf_counter() >= deadline:
            trial["error"] = {
                "type": "timeout",
                "message": "request exceeded the {:.3g}s overall timeout".format(timeout_seconds),
            }
        else:
            trial["error"] = {"type": "request_error", "message": str(exc)}
    except Exception as exc:  # Preserve unexpected per-trial failures in machine-readable output.
        if deadline_expired.is_set() or time.perf_counter() >= deadline:
            trial["error"] = {
                "type": "timeout",
                "message": "request exceeded the {:.3g}s overall timeout".format(timeout_seconds),
            }
        else:
            trial["error"] = {
                "type": "client_error",
                "message": "{}: {}".format(type(exc).__name__, exc),
            }
    finally:
        trial["wall_time_ms"] = round((time.perf_counter() - started) * 1000, 3)
        if deadline_timer is not None:
            deadline_timer.cancel()
        if response is not None:
            response.close()
        if connection is not None:
            connection.close()
    return trial


def build_schedule(repeats, include_warmup, prompt_set="general"):
    config = PROMPT_SETS.get(prompt_set, PROMPT_SETS["general"])
    schedule = []
    if include_warmup:
        schedule.append(("warmup", "warmup", config["warmup"]))
    for _ in range(repeats):
        schedule.append(("repeat", "repeat", config["repeat"]))
    for prompt_id, prompt in config["unrelated"]:
        schedule.append(("unrelated", prompt_id, prompt))
    return schedule


def run_benchmark(url, requested_model="default", repeats=3, include_warmup=False,
                  timeout_seconds=60.0, api_key=None, api_key_env=None,
                  max_tokens=256, prompt_set="general"):
    run_started = _utc_now()
    trials = []
    config = PROMPT_SETS.get(prompt_set, PROMPT_SETS["general"])
    for index, (kind, prompt_id, prompt) in enumerate(build_schedule(repeats, include_warmup, prompt_set), 1):
        trials.append(
            run_trial(
                url=url,
                prompt=prompt,
                prompt_id=prompt_id,
                kind=kind,
                index=index,
                requested_model=requested_model,
                timeout_seconds=timeout_seconds,
                api_key=api_key,
                max_tokens=max_tokens,
            )
        )

    failed = sum(1 for trial in trials if trial["status"] != "ok")
    model_identities = []
    server_headers = []
    for trial in trials:
        for model in trial["model_identities"]:
            if model not in model_identities:
                model_identities.append(model)
        server_header = trial["server_identity"]["server_header"]
        if server_header and server_header not in server_headers:
            server_headers.append(server_header)

    return {
        "schema_version": 1,
        "client": {
            "name": "benchmark-ps5lm",
            "version": CLIENT_VERSION,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "run": {
            "started_at_utc": run_started,
            "endpoint": _safe_endpoint(url),
            "requested_model": requested_model,
            "timeout_seconds": timeout_seconds,
            "max_tokens": max_tokens,
            "repeat_count": repeats,
            "warmup_included": include_warmup,
            "prompt_set": config["id"],
            "prompt_category": prompt_set,
            "unrelated_prompt_ids": [prompt_id for prompt_id, _ in config["unrelated"]],
            "request_parameters": request_parameters(max_tokens),
            "api_key_environment_variable_used": bool(api_key),
            "api_key_environment_variable": api_key_env,
        },
        "server_identity": {
            "model_identities_reported": model_identities,
            "server_headers_reported": server_headers,
        },
        "summary": {
            "trial_count": len(trials),
            "failed_trial_count": failed,
            "success": failed == 0,
        },
        "trials": trials,
    }


def _positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return number


def _positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _write_json(path, value):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary_path = path + ".tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8", newline="\n") as output:
            json.dump(value, output, indent=2, ensure_ascii=False)
            output.write("\n")
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def make_parser():
    parser = argparse.ArgumentParser(
        description="Measure client-visible latency for fixed prompts against a running OpenAI SSE endpoint."
    )
    parser.add_argument(
        "--url",
        required=True,
        help="full URL of the already-running streaming chat completions endpoint",
    )
    parser.add_argument("--model", default="default", help="request model label (default: default)")
    parser.add_argument("--repeats", type=_positive_int, default=3, help="identical prompt trials (default: 3)")
    parser.add_argument("--max-tokens", type=_positive_int, default=256, help="server generation limit (default: 256)")
    parser.add_argument("--warmup", action="store_true", help="run one separate warmup trial before measured trials")
    parser.add_argument("--timeout", type=_positive_float, default=60.0, help="overall timeout per trial in seconds")
    parser.add_argument(
        "--prompt-set",
        choices=["general", "code", "prose"],
        default="general",
        help="prompt category to benchmark (default: general)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("build", "benchmark-results.json"),
        help="raw JSON result path (default: build/benchmark-results.json)",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        metavar="NAME",
        help="explicitly opt in to reading a bearer token from environment variable NAME",
    )
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        _parse_endpoint(args.url)
    except ValueError as exc:
        parser.error(str(exc))

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    report = run_benchmark(
        url=args.url,
        requested_model=args.model,
        repeats=args.repeats,
        include_warmup=args.warmup,
        timeout_seconds=args.timeout,
        api_key=api_key,
        api_key_env=args.api_key_env,
        max_tokens=args.max_tokens,
        prompt_set=args.prompt_set,
    )
    try:
        _write_json(args.output, report)
    except OSError as exc:
        print("Could not write result JSON to {}: {}".format(args.output, exc), file=sys.stderr)
        return 2

    summary = report["summary"]
    print(
        "Wrote {} trials ({} failed) to {}".format(
            summary["trial_count"], summary["failed_trial_count"], args.output
        )
    )
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
