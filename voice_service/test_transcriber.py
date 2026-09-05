"""
Unit tests for transcriber.py. These mock the WhisperModel so tests run
fast, offline, and without downloading model weights — good for CI.

Run with:
    pytest test_transcriber.py -v
"""

from unittest.mock import MagicMock, patch

from transcriber import Transcriber, TranscriptionResult


def _fake_segment(text, start, end):
    seg = MagicMock()
    seg.text = text
    seg.start = start
    seg.end = end
    return seg


def _fake_info(language="ar", probability=0.97):
    info = MagicMock()
    info.language = language
    info.language_probability = probability
    return info


def test_transcribe_file_normal_case():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (
        [_fake_segment(" ابني مش بينام كويس بالليل ", 0.0, 3.2)],
        _fake_info(),
    )
    t = Transcriber(mock_model)

    result = t.transcribe_file("fake_path.wav")

    assert isinstance(result, TranscriptionResult)
    assert result.text == "ابني مش بينام كويس بالليل"
    assert result.warning is None
    assert result.language == "ar"
    assert result.duration_seconds == 3.2


def test_transcribe_file_no_speech_detected():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], _fake_info())
    t = Transcriber(mock_model)

    result = t.transcribe_file("silent.wav")

    assert result.text == ""
    assert result.warning == "no_speech_detected"


def test_transcribe_file_clip_too_short():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (
        [_fake_segment("اه", 0.0, 0.15)],
        _fake_info(),
    )
    t = Transcriber(mock_model)

    with patch("config.MIN_AUDIO_SECONDS", 0.4):
        result = t.transcribe_file("tooshort.wav")

    assert result.text == ""
    assert result.warning == "clip_too_short"


def test_transcribe_file_multiple_segments_joined():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (
        [
            _fake_segment(" ابني عنده تلات سنين ", 0.0, 2.0),
            _fake_segment(" وبيرفض ياكل خضار ", 2.0, 4.5),
        ],
        _fake_info(),
    )
    t = Transcriber(mock_model)

    result = t.transcribe_file("fake.wav")

    assert result.text == "ابني عنده تلات سنين وبيرفض ياكل خضار"
    assert result.duration_seconds == 4.5


def test_transcribe_bytes_cleans_up_temp_file():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (
        [_fake_segment("تجربة", 0.0, 1.0)],
        _fake_info(),
    )
    t = Transcriber(mock_model)

    with patch("os.remove") as mock_remove:
        t.transcribe_bytes(b"fake audio bytes", suffix=".webm")
        assert mock_remove.called  # temp file must always be cleaned up


def test_transcribe_bytes_cleans_up_even_on_failure():
    mock_model = MagicMock()
    mock_model.transcribe.side_effect = RuntimeError("corrupt audio")
    t = Transcriber(mock_model)

    with patch("os.remove") as mock_remove:
        try:
            t.transcribe_bytes(b"garbage", suffix=".webm")
        except RuntimeError:
            pass
        assert mock_remove.called  # cleanup must happen even when transcription raises
