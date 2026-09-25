"""Exercise the standalone preparation scripts without network access."""

import io
import pickle
import runpy
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(params=["shakespeare", "shakespeare_char"])
def prepare_script(request, tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "data" / request.param / "prepare.py"
    script = tmp_path / "prepare.py"
    shutil.copyfile(source, script)
    # Catch accidental direct requests imports, even when it is installed transitively.
    monkeypatch.setitem(sys.modules, "requests", None)
    return script


@pytest.fixture
def encoded_text(monkeypatch):
    chunks = []

    def encode_ordinary(text):
        chunks.append(text)
        return list(text.encode("utf-8"))

    def get_encoding(name):
        assert name == "gpt2"
        return SimpleNamespace(encode_ordinary=encode_ordinary)

    # Tokenizer internals are independent of the download; keep this test offline.
    monkeypatch.setitem(sys.modules, "tiktoken", SimpleNamespace(get_encoding=get_encoding))
    return chunks


@pytest.mark.parametrize("cached", [False, True])
def test_prepare_preserves_text_and_token_files(prepare_script, encoded_text, monkeypatch, cached):
    raw_text = "To bé,\r\nor not.\n"
    text = "To bé,\nor not.\n"
    input_path = prepare_script.parent / "input.txt"
    response = io.BytesIO(raw_text.encode("utf-8"))

    def urlopen(url):
        assert not cached, "cached input must not trigger a download"
        assert url == (
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        )
        return response

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    if cached:
        input_path.write_bytes(raw_text.encode("utf-8"))

    runpy.run_path(str(prepare_script))

    assert input_path.read_text(encoding="utf-8") == text
    if not cached:
        assert response.closed
    train = np.fromfile(prepare_script.parent / "train.bin", dtype=np.uint16)
    val = np.fromfile(prepare_script.parent / "val.bin", dtype=np.uint16)
    if encoded_text:
        assert encoded_text == ["To bé,\nor not", ".\n"]
        assert train.tolist() == list("To bé,\nor not".encode("utf-8"))
        assert val.tolist() == list(".\n".encode("utf-8"))
    else:
        with (prepare_script.parent / "meta.pkl").open("rb") as f:
            meta = pickle.load(f)
        assert meta["vocab_size"] == len(set(text))
        assert "".join(meta["itos"][int(i)] for i in train) == "To bé,\nor not"
        assert "".join(meta["itos"][int(i)] for i in val) == ".\n"
        assert all(meta["itos"][i] == ch for ch, i in meta["stoi"].items())


@pytest.mark.parametrize("failure", ["http", "connection", "decode", "read"])
def test_failed_download_does_not_leave_cached_input(prepare_script, monkeypatch, failure):
    class BrokenResponse(io.BytesIO):
        def read(self):
            raise OSError("interrupted download")

    def urlopen(url):
        if failure == "http":
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        if failure == "connection":
            raise urllib.error.URLError("connection failed")
        if failure == "read":
            return BrokenResponse()
        return io.BytesIO(b"\xff")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with pytest.raises((OSError, UnicodeDecodeError)):
        runpy.run_path(str(prepare_script))
    for filename in ("input.txt", "train.bin", "val.bin", "meta.pkl"):
        assert not (prepare_script.parent / filename).exists()
