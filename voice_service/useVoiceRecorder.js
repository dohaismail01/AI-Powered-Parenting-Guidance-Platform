// Minimal React hook: record mic audio, send to the /stt FastAPI endpoint,
// return the Arabic transcript. Pair with stt_service.py.
//
// Usage in a component:
//   const { isRecording, startRecording, stopRecording, transcript, loading } = useVoiceRecorder();

import { useState, useRef } from "react";

const STT_ENDPOINT = "http://localhost:8001/stt"; // change to your deployed URL

export function useVoiceRecorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [loading, setLoading] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [error, setError] = useState(null);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  async function startRecording() {
    setError(null);
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    chunksRef.current = [];

    recorder.ondataavailable = (e) => chunksRef.current.push(e.data);
    recorder.onstop = handleStop;

    recorder.start();
    mediaRecorderRef.current = recorder;
    setIsRecording(true);
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
    mediaRecorderRef.current?.stream.getTracks().forEach((t) => t.stop());
    setIsRecording(false);
  }

  async function handleStop() {
    setLoading(true);
    try {
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      const formData = new FormData();
      formData.append("audio", blob, "recording.webm");

      const res = await fetch(STT_ENDPOINT, { method: "POST", body: formData });
      if (!res.ok) throw new Error(`STT request failed: ${res.status}`);

      const data = await res.json();
      if (data.warning === "no_speech_detected" || data.warning === "clip_too_short") {
        setError("لم يتم رصد كلام واضح، حاول تاني"); // "No clear speech detected, try again"
        setTranscript("");
      } else {
        setTranscript(data.text);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return { isRecording, startRecording, stopRecording, transcript, loading, error };
}
