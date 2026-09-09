import { useEffect, useRef, useState } from "react";
import { useVoiceRecorder } from "./useVoiceRecorder.js";

// API paths are proxied to the FastAPI service (:8001) by vite.config.js,
// so we can use same-origin relative URLs here.
async function getHealth() {
  const r = await fetch("/health");
  if (!r.ok) throw new Error("health failed");
  return r.json();
}

// One-tap starter questions so users don't have to type Arabic.
const EXAMPLES = [
  "ابني عنده ٥ سنين وبيعمل نوبات غضب، أعمل إيه؟",
  "بنتي عندها سنتين مش عايزة تنام بالليل",
  "إزاي أعلّم ابني يسمع الكلام من غير ضرب؟",
  "قد إيه وقت الموبايل المناسب لطفل عنده ٤ سنين؟",
];

function StatusDots({ health }) {
  const dot = (on) => ({
    display: "inline-block",
    width: 8,
    height: 8,
    borderRadius: "50%",
    marginInlineEnd: 5,
    verticalAlign: "middle",
    background: on ? "var(--ok)" : "var(--danger)",
  });
  return (
    <div className="status">
      <span><span style={dot(health?.status === "ok")} />التعرف على الصوت</span>
      <span><span style={dot(health?.rag_available)} />محرّك الإجابات</span>
      <span><span style={dot(health?.tts_available)} />النطق</span>
    </div>
  );
}

function Sources({ items }) {
  if (!items || !items.length) return null;
  return (
    <div className="sources">
      <b>المصادر:</b> {items.join(" · ")}
    </div>
  );
}

function Message({ msg }) {
  return (
    <div className={"msg " + msg.role}>
      <div className="who">{msg.role === "user" ? (msg.voice ? "أنت (صوت)" : "أنت") : "ParentWise"}</div>
      {msg.role === "user" ? (
        <div className="body">{msg.text}</div>
      ) : msg.pending ? (
        <div className="body">
          <span className="spinner" /> بيجهّز إجابة مدعومة بالمصادر… <b>{msg.elapsed} ث</b>
          {msg.elapsed >= 8 && (
            <div className="hint" style={{ textAlign: "right", marginTop: 4 }}>
              على الجهاز ده الإجابة ممكن تاخد دقيقة أو اتنين — استنّى شوية 🙏
            </div>
          )}
        </div>
      ) : msg.error ? (
        <div className="err">⚠️ {msg.error}</div>
      ) : (
        <div className="body">
          {msg.badges?.length ? (
            <div style={{ marginBottom: 6 }}>
              {msg.badges.map((b, i) => (
                <span key={i} className={"badge " + b.kind}>{b.text}</span>
              ))}
            </div>
          ) : null}
          {msg.showText && <div>{msg.text || "(لا توجد إجابة)"}</div>}
          {!msg.showText && <div className="hint" style={{ textAlign: "right" }}>(الرد بالصوت فقط)</div>}
          <Sources items={msg.sources} />
          {msg.audio && (
            <audio controls autoPlay src={`data:${msg.audioMime || "audio/wav"};base64,${msg.audio}`} />
          )}
          {!msg.audio && msg.audioError && (
            <div className="hint" style={{ textAlign: "right" }}>⚠️ تعذّر إنشاء الصوت — {msg.audioError}</div>
          )}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [messages, setMessages] = useState([]);
  const [text, setText] = useState("");
  const [outText, setOutText] = useState(true);
  const [outVoice, setOutVoice] = useState(true);
  const [busy, setBusy] = useState(false);
  const chatRef = useRef(null);
  const timersRef = useRef({});

  useEffect(() => {
    let alive = true;
    const tick = () => getHealth().then((h) => alive && setHealth(h)).catch(() => alive && setHealth(null));
    tick();
    const id = setInterval(tick, 15000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  useEffect(() => {
    if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, [messages]);

  function updateMsg(id, patch) {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch } : m)));
  }

  async function submit({ text: q = null, audioBlob = null }) {
    if (busy) return;
    const speak = outVoice;
    const showText = outText || !outVoice; // never render an empty bubble
    setBusy(true);
    setText(""); // clear the input immediately, not after the (slow) response

    const userId = Date.now() + "-u";
    const botId = Date.now() + "-b";
    setMessages((prev) => [
      ...prev,
      { id: userId, role: "user", text: q || "🎤 (رسالة صوتية)", voice: !!audioBlob },
      { id: botId, role: "bot", pending: true, elapsed: 0 },
    ]);

    const started = Date.now();
    timersRef.current[botId] = setInterval(() => {
      updateMsg(botId, { elapsed: Math.floor((Date.now() - started) / 1000) });
    }, 1000);

    const fd = new FormData();
    if (q) fd.append("text", q);
    if (audioBlob) fd.append("audio", audioBlob, "input.webm");
    fd.append("speak", speak ? "true" : "false");

    try {
      const r = await fetch("/ask_voice", { method: "POST", body: fd });
      clearInterval(timersRef.current[botId]);
      if (!r.ok) {
        let d = {};
        try { d = await r.json(); } catch { /* non-json error body */ }
        updateMsg(botId, { pending: false, error: d.detail || `خطأ ${r.status}` });
      } else {
        const data = await r.json();
        if (audioBlob && data.transcript) {
          updateMsg(userId, { text: data.transcript });
        }
        const badges = [];
        if (data.is_high_risk) badges.push({ kind: "risk", text: "تحويل لمختص" });
        if (data.is_grounded === false) badges.push({ kind: "note", text: "تحقّق من مصدر إضافي" });
        updateMsg(botId, {
          pending: false,
          showText,
          text: data.answer,
          sources: data.sources,
          badges,
          audio: speak ? data.audio : null,
          audioMime: data.audio_mime,
          audioError: data.audio_error,
        });
      }
    } catch (e) {
      clearInterval(timersRef.current[botId]);
      updateMsg(botId, { pending: false, error: "تعذّر الاتصال بالخادم" });
    } finally {
      setBusy(false);
      setText("");
    }
  }

  const recorder = useVoiceRecorder((blob) => submit({ audioBlob: blob }));

  function onSend() {
    const q = text.trim();
    if (q) submit({ text: q });
  }
  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); onSend(); }
  }

  return (
    <div className="app">
      <header>
        <h1>👶 ParentWise</h1>
        <p>مساعد ذكي للأهل — اسأل بالكلام أو بالكتابة</p>
        <StatusDots health={health} />
      </header>

      <div className="chat" ref={chatRef}>
        {messages.length === 0 ? (
          <div className="empty">
            <div className="empty-lead">👋 اسأل عن أي حاجة في تربية طفلك — بالكتابة أو بالصوت.</div>
            <div className="empty-sub">تقدر تجرّب سؤال من دول:</div>
            <div className="chips">
              {EXAMPLES.map((q, i) => (
                <button key={i} className="chip" onClick={() => submit({ text: q })} disabled={busy}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m) => <Message key={m.id} msg={m} />)
        )}
      </div>

      <div className="controls">
        <div className="outrow">
          <span>طريقة الرد:</span>
          <label><input type="checkbox" checked={outText} onChange={(e) => setOutText(e.target.checked)} /> نص</label>
          <label><input type="checkbox" checked={outVoice} onChange={(e) => setOutVoice(e.target.checked)} /> صوت</label>
          {messages.length > 0 && (
            <button className="clear" onClick={() => setMessages([])} disabled={busy} style={{ marginInlineStart: "auto" }}>
              🗑 مسح المحادثة
            </button>
          )}
        </div>
        <div className="inrow">
          <button
            className={"mic" + (recorder.isRecording ? " rec" : "")}
            onClick={() => (recorder.isRecording ? recorder.stopRecording() : recorder.startRecording())}
            disabled={busy && !recorder.isRecording}
            title="اضغط للتسجيل"
          >
            {recorder.isRecording ? "⏹" : "🎤"}
          </button>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="اكتب سؤالك هنا…"
            rows={1}
            disabled={busy}
          />
          <button className="send" onClick={onSend} disabled={busy || !text.trim()}>إرسال</button>
        </div>
        <div className="hint">
          {recorder.isRecording
            ? "بيسجّل… اضغط لإيقاف التسجيل والإرسال"
            : recorder.error || "Enter للإرسال · Shift+Enter لسطر جديد"}
        </div>
      </div>
    </div>
  );
}
