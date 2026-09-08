// React hook: record mic audio and hand the resulting webm Blob to a callback.
// (Adapted from voice_service/useVoiceRecorder.js, which posted to /stt; here
// the App sends the blob to /ask_voice instead so one call does STT+RAG+TTS.)
import { useState, useRef } from "react";

export function useVoiceRecorder(onRecorded) {
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState(null);
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);

  async function startRecording() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        if (blob.size > 0 && onRecorded) onRecorded(blob);
      };
      recorder.start();
      recorderRef.current = recorder;
      setIsRecording(true);
    } catch (e) {
      setError(
        "تعذّر الوصول للميكروفون — تأكد من الإذن ومن فتح الصفحة عبر localhost"
      );
    }
  }

  function stopRecording() {
    const r = recorderRef.current;
    if (r && r.state !== "inactive") r.stop();
    setIsRecording(false);
  }

  return { isRecording, error, startRecording, stopRecording };
}
