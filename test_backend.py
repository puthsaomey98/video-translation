import os
import tempfile
import pytest
from fastapi.testclient import TestClient
from backend import app, parse_srt

client = TestClient(app)

def test_parse_srt_valid():
    """Test that the SRT parser accurately converts timestamps and text."""
    srt_content = """1
00:00:01,000 --> 00:00:04,500
Hello world from subtitle!

2
00:00:05,100 --> 00:00:09,800
Second sentence test.
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False, encoding="utf-8") as f:
        f.write(srt_content)
        temp_path = f.name

    try:
        segments = parse_srt(temp_path)
        assert len(segments) == 2
        assert segments[0]["start"] == 1.0
        assert segments[0]["end"] == 4.5
        assert segments[0]["text"] == "Hello world from subtitle!"
        
        assert segments[1]["start"] == 5.1
        assert segments[1]["end"] == 9.8
        assert segments[1]["text"] == "Second sentence test."
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_parse_srt_webvtt():
    """Test that the parser handles WebVTT (.vtt) headers properly."""
    vtt_content = """WEBVTT

1
00:00:02,000 --> 00:00:06,000
WebVTT test line.
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vtt", delete=False, encoding="utf-8") as f:
        f.write(vtt_content)
        temp_path = f.name

    try:
        segments = parse_srt(temp_path)
        assert len(segments) == 1
        assert segments[0]["start"] == 2.0
        assert segments[0]["end"] == 6.0
        assert segments[0]["text"] == "WebVTT test line."
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_api_input_files():
    """Test the input files API endpoint."""
    response = client.get("/api/input-files")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_api_input_subtitles():
    """Test the input subtitles API endpoint."""
    response = client.get("/api/input-subtitles")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_api_history():
    """Test the project history API endpoint."""
    response = client.get("/api/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_status_not_found():
    """Test querying a non-existent task status returns 404."""
    response = client.get("/api/status/invalid_task_id_999")
    assert response.status_code == 404