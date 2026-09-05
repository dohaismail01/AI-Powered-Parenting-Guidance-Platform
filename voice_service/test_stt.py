"""
Quick manual check: transcribe a single local audio file and print the
result. Use this to sanity-test accuracy on real Egyptian-dialect
recordings before wiring anything into the FastAPI service.

Usage:
    python test_stt.py path/to/audio.wav
"""

import sys

from transcriber import Transcriber


def main():
    if len(sys.argv) != 2:
        print("Usage: python test_stt.py path/to/audio.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    t = Transcriber.load()
    result = t.transcribe_file(audio_path)

    print("\n--- Result ---")
    print(f"Language: {result.language} (confidence: {result.language_confidence:.2f})")
    print(f"Audio duration: {result.duration_seconds:.2f}s")
    print(f"Processing time: {result.processing_time_seconds:.2f}s")
    if result.warning:
        print(f"Warning: {result.warning}")
    print(f"\nTranscript:\n{result.text}")


if __name__ == "__main__":
    main()
