import { useState, useEffect, useRef, useCallback } from "react";

// ─── CONFIG ───────────────────────────────────────────────────
const GEMINI_KEY = import.meta.env.VITE_GEMINI_API_KEY;
const MODELS = { flash: "gemini-2.5-flash", pro: "gemini-2.5-flash" };
const VALID_USER = { id: "test_01", pass: "test@1234", name: "Arjun Kapoor", role: "Senior Sales Executive" };

// ─── CASCADEFLOW ──────────────────────────────────────────────
function cascadeRoute(msg) {
  const m = msg.toLowerCase();
  const complex = msg.length > 250 ||
    /proposal|contract|draft|generate|write|email|report|summar|analyz|strategy|negoti|discount|offer|objection|competitor|compare|recommend|plan|quote|pricing/.test(m);
  return complex
    ? { model: MODELS.pro,   tier: "Pro",   reason: "Complex task — advanced reasoning needed" }
    : { model: MODELS.flash, tier: "Flash", reason: "Simple query — lightweight model sufficient" };
}
function detectTask(msg) {
  const m = msg.toLowerCase();
  if (/email|follow.?up/.test(m))         return "follow_up_email";
  if (/proposal|quote|pricing/.test(m))   return "proposal";
  if (/summar|recap/.test(m))             return "meeting_summary";
  if (/objection|concern|compet/.test(m)) return "objection_handling";
  if (/discount|offer|deal/.test(m))      return "discount_suggestion";
  if (/strategy|approach|tactic/.test(m)) return "negotiation_strategy";
  if (/contract|agreement/.test(m))       return "contract";
  return "conversation";
}

// ─── GEMINI API ───────────────────────────────────────────────
async function callGemini({ model, systemPrompt, history, userMessage, onChunk }) {
  console.log("Model:", model);
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${GEMINI_KEY}`;
  const contents = [
    ...(history || []).map(m => ({ role: m.role === "assistant" ? "model" : "user", parts: [{ text: m.content }] })),
    { role: "user", parts: [{ text: userMessage }] },
  ];
  const t0 = performance.now();
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      system_instruction: { parts: [{ text: systemPrompt }] },
      contents,
      generationConfig: { temperature: 0.75, maxOutputTokens: 2048, topP: 0.95 },
      safetySettings: [
        { category: "HARM_CATEGORY_HARASSMENT",        threshold: "BLOCK_NONE" },
        { category: "HARM_CATEGORY_HATE_SPEECH",       threshold: "BLOCK_NONE" },
        { category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold: "BLOCK_NONE" },
        { category: "HARM_CATEGORY_DANGEROUS_CONTENT", threshold: "BLOCK_NONE" },
      ],
    }),
  });
  if (!res.ok) {
    const e = await res.json().catch(() => ({}));
    throw new Error(e?.error?.message || `HTTP ${res.status}`);
  }
  const data = await res.json();
  const lat = Math.round(performance.now() - t0);
  const full = data?.candidates?.[0]?.content?.parts?.[0]?.text || "";
  if (!full) {
    const reason = data?.candidates?.[0]?.finishReason;
    if (reason === "SAFETY") throw new Error("Blocked by safety filter. Try rephrasing.");
    throw new Error("Empty response. Check API key permissions at aistudio.google.com");
  }
  if (onChunk) {
    const words = full.split(" ");
    let acc = "";
    for (let i = 0; i < words.length; i++) {
      acc += (i === 0 ? "" : " ") + words[i];
      onChunk(acc);
      if (i % 8 === 0) await new Promise(r => setTimeout(r, 15));
    }
  }
  const inTok  = data?.usageMetadata?.promptTokenCount     || Math.round((systemPrompt + userMessage).length / 4);
  const outTok = data?.usageMetadata?.candidatesTokenCount || Math.round(full.length / 4);
  const tok = inTok + outTok;
  const costINR = (tok / 1000) * (model === MODELS.pro ? 0.00375 : 0.000075) * 83;
  return { text: full, latency: lat, inputTokens: inTok, outputTokens: outTok, totalTokens: tok, costINR, model };
}

// ─── HINDSIGHT MEMORY ─────────────────────────────────────────
const HS = {
  load: (id) => { try { return JSON.parse(localStorage.getItem(`hs_${id}`)) || []; } catch { return []; } },
  save: (id, m) => { try { localStorage.setItem(`hs_${id}`, JSON.stringify(m)); } catch {} },
  add:  (id, mem) => { const m = [mem, ...HS.load(id)].slice(0, 100); HS.save(id, m); return m; },
  ctx:  (id) => {
    const m = HS.load(id);
    if (!m.length) return "";
    return m.slice(0, 20).map(x => `[${x.type.toUpperCase()} — ${new Date(x.date).toLocaleDateString("en-IN")}]: ${x.content}`).join("\n");
  },
};

// ─── LOCAL DB ─────────────────────────────────────────────────
function dbGet(k) { try { return JSON.parse(localStorage.getItem(`nai_${k}`)); } catch { return null; } }
function dbSet(k, v) { try { localStorage.setItem(`nai_${k}`, JSON.stringify(v)); } catch {} }
function useDB(key, init) {
  const [s, ss] = useState(() => dbGet(key) ?? init);
  const set = useCallback((v) => ss(p => { const n = typeof v === "function" ? v(p) : v; dbSet(key, n); return n; }), [key]);
  return [s, set];
}

// ─── SYSTEM PROMPT ────────────────────────────────────────────
function buildSP(c) {
  const mem = HS.ctx(c.id);
  return `You are NegotiAI — an elite AI sales negotiation assistant for an Indian B2B SaaS company.

== CUSTOMER PROFILE ==
Name: ${c.name} | Company: ${c.company} | City: ${c.city || ""} | Industry: ${c.industry || ""}
Budget: ${c.budget || "Not disclosed"} | Status: ${c.status || "active"}
Requirements: ${c.requirements || "Not specified"}
Known Objections: ${c.objections || "None"}
Competitors: ${c.competitors || "None"}
Communication Style: ${c.style || "formal"}

== HINDSIGHT MEMORY ==
${mem || "No previous interactions — fresh conversation."}

== INSTRUCTIONS ==
- Use Indian context: amounts in ₹, reference DPDP Act, GST, SEBI, NMC, ABDM where relevant.
- NEVER ask for info already in the profile or memory above.
- Reference past interactions naturally ("As we discussed...", "Based on your concern about DPDP...").
- For proposals: provide line-item pricing in ₹ with payment milestones.
- For objections: acknowledge, validate, then counter with data.
- For emails: write complete copy ready to send.
- For strategy: give 3 ranked tactics with rationale.
- End every response with "⚡ Recommended Next Action" unless it is a trivial factual answer.
- Use **bold headers** to organise responses longer than 3 sentences.`;
}

// ─── SEED DATA ────────────────────────────────────────────────
const SEED = [
  { id: "c1", name: "Rajesh Mehta", company: "TechBridge Solutions Pvt Ltd", email: "rajesh@techbridge.in", phone: "+91-9876543210", city: "Mumbai", industry: "IT Services", budget: "₹8,00,000 – ₹12,00,000", requirements: "Enterprise CRM with mobile app, Hindi UI, DPDP Act compliance, Tally ERP integration", objections: "High upfront cost. DPDP data localisation concern. Zoho offered cheaper.", competitors: "Zoho CRM (₹7.5L), Salesforce India (₹14L)", style: "formal", priority: "high", status: "negotiating", createdAt: new Date(Date.now() - 12 * 86400000).toISOString() },
  { id: "c2", name: "Priya Nair", company: "FreshMart Retail Chains", email: "priya@freshmart.in", phone: "+91-9845001122", city: "Bengaluru", industry: "Retail/FMCG", budget: "₹3,00,000 – ₹5,00,000", requirements: "Inventory management with GST auto-filing, WhatsApp alerts, multi-store dashboard", objections: "Small IT team, worried about implementation time and support cost", competitors: "Unicommerce, Vyapar App", style: "casual", priority: "medium", status: "discovery", createdAt: new Date(Date.now() - 5 * 86400000).toISOString() },
  { id: "c3", name: "Dr. Amit Verma", company: "HealthFirst Hospital Network", email: "a.verma@healthfirst.in", phone: "+91-9711223344", city: "Delhi", industry: "Healthcare", budget: "₹18,00,000 – ₹25,00,000", requirements: "HMIS with ABDM integration, insurance claim automation, NMC compliance, 99.9% SLA, 3 locations", objections: "Must have NMC & ABDM cert. Previous vendor missed SLA. Needs escrow payment.", competitors: "Practo Enterprise, HIS by Insta", style: "technical", priority: "high", status: "proposal", createdAt: new Date(Date.now() - 20 * 86400000).toISOString() },
];

// ─── THEME ────────────────────────────────────────────────────
const T = {
  bg: "#07071A", surface: "#0C0C22", card: "#10102A",
  border: "rgba(120,100,255,0.12)", borderMed: "rgba(140,120,255,0.22)",
  accent: "#7C6FFF", accentDim: "rgba(124,111,255,0.13)", accentBd: "rgba(124,111,255,0.32)",
  accent2: "#06D6FE", accent2Dim: "rgba(6,214,254,0.10)",
  purple: "#BF5FFF", purpleDim: "rgba(191,95,255,0.12)",
  green: "#00E5A0", greenDim: "rgba(0,229,160,0.10)",
  amber: "#FFB547", amberDim: "rgba(255,181,71,0.10)",
  red: "#FF5B6B", redDim: "rgba(255,91,107,0.10)",
  blue: "#4DA6FF", blueDim: "rgba(77,166,255,0.10)",
  pink: "#FF4DC4", pinkDim: "rgba(255,77,196,0.10)",
  text: "#E8E6FF", sub: "#9896C8", muted: "#4E4C7A",
};

// ─── GLOBAL STYLES ────────────────────────────────────────────
const GS = () => (
  <style>{`
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@700;800&display=swap');
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html, body, #root { height: 100%; }
    body { background: ${T.bg}; color: ${T.text}; font-family: 'Inter', -apple-system, sans-serif; font-size: 14px; line-height: 1.6; -webkit-font-smoothing: antialiased; }
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(124,111,255,0.2); border-radius: 10px; }

    @keyframes fadeUp    { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:translateY(0); } }
    @keyframes fadeIn    { from { opacity:0; } to { opacity:1; } }
    @keyframes slideR    { from { opacity:0; transform:translateX(-8px); } to { opacity:1; transform:translateX(0); } }
    @keyframes spin      { to { transform: rotate(360deg); } }
    @keyframes blink     { 0%,100%{opacity:1} 50%{opacity:0} }
    @keyframes float3d   { 0%,100%{transform:translateY(0px) rotateX(0deg);} 50%{transform:translateY(-6px) rotateX(3deg);} }
    @keyframes pulseGlow { 0%,100%{box-shadow:0 0 18px rgba(124,111,255,0.4),0 0 40px rgba(124,111,255,0.15);} 50%{box-shadow:0 0 28px rgba(124,111,255,0.6),0 0 60px rgba(124,111,255,0.25);} }
    @keyframes shimmer   { 0%{background-position:-200% center;} 100%{background-position:200% center;} }
    @keyframes orbitSpin { to { transform: rotate(360deg); } }
    @keyframes scaleIn   { from{opacity:0;transform:scale(0.88);} to{opacity:1;transform:scale(1);} }
    @keyframes gradShift { 0%,100%{background-position:0% 50%;} 50%{background-position:100% 50%;} }
    @keyframes ripple    { 0%{transform:scale(0);opacity:0.6;} 100%{transform:scale(2.5);opacity:0;} }
    @keyframes slideUp   { from{opacity:0;transform:translateY(20px);} to{opacity:1;transform:translateY(0);} }
    @keyframes bgPulse   { 0%,100%{opacity:0.4;} 50%{opacity:0.7;} }
    @keyframes loadProg  { from{width:0%;} to{width:100%;} }
    @keyframes particleFly { 0%{opacity:0;transform:translateY(0) scale(0);} 20%{opacity:1;} 100%{opacity:0;transform:translateY(-120px) scale(1);} }
    @keyframes hexSpin   { to { transform: rotate(360deg) scale(1.05); } }
    @keyframes typewriter{ from{width:0;} to{width:100%;} }
    @keyframes numberCount{ from{opacity:0;transform:translateY(4px);} to{opacity:1;transform:translateY(0);} }
    @keyframes borderGlow{ 0%,100%{border-color:rgba(124,111,255,0.3);} 50%{border-color:rgba(124,111,255,0.7);} }
    @keyframes neonPulse { 0%,100%{text-shadow:0 0 7px ${T.accent},0 0 20px ${T.accent};} 50%{text-shadow:0 0 14px ${T.accent},0 0 40px ${T.accent},0 0 80px ${T.accent}55;} }
    @keyframes dash { to { stroke-dashoffset: 0; } }
    @keyframes ping  { 75%,100% { transform: scale(2); opacity: 0; } }

    .au  { animation: fadeUp .28s cubic-bezier(.22,1,.36,1) both; }
    .ar  { animation: slideR .2s ease both; }
    .ai  { animation: fadeIn .3s ease both; }
    .asi { animation: scaleIn .32s cubic-bezier(.22,1,.36,1) both; }
    .spin { animation: spin 1s linear infinite; }
    .tc::after { content: '▋'; animation: blink .65s step-end infinite; color: ${T.accent}; }
    .float { animation: float3d 4s ease-in-out infinite; }
    .pulse-glow { animation: pulseGlow 2.5s ease-in-out infinite; }
    .neon { animation: neonPulse 2s ease-in-out infinite; }

    .card {
      background: ${T.card};
      border: 1px solid ${T.border};
      border-radius: 14px;
      transition: border-color .2s, box-shadow .2s, transform .2s;
    }
    .card-3d {
      transform-style: preserve-3d;
      perspective: 800px;
      transition: transform .3s cubic-bezier(.22,1,.36,1), box-shadow .3s;
    }
    .card-3d:hover {
      transform: translateY(-4px) rotateX(2deg);
      box-shadow: 0 16px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(124,111,255,0.25);
    }
    .ch:hover { border-color: ${T.borderMed}; background: rgba(124,111,255,0.04); }

    .tag { display:inline-flex; align-items:center; gap:4px; padding:2px 9px; border-radius:6px; font-size:11px; font-weight:600; letter-spacing:.03em; }

    .btn {
      display:inline-flex; align-items:center; gap:7px;
      padding:9px 16px; border-radius:10px; font-size:13px; font-weight:600;
      border:none; cursor:pointer;
      transition: all .18s cubic-bezier(.22,1,.36,1);
      white-space:nowrap; position:relative; overflow:hidden;
    }
    .btn::after {
      content:''; position:absolute; inset:0; opacity:0;
      background:radial-gradient(circle at center, rgba(255,255,255,0.18) 0%, transparent 70%);
      transition: opacity .15s;
    }
    .btn:active::after { opacity:1; }
    .btn:active { transform: scale(0.96) translateY(1px); }

    .bp {
      background: linear-gradient(135deg, ${T.accent}, ${T.purple});
      color: #fff;
      box-shadow: 0 4px 18px rgba(124,111,255,0.35), 0 1px 0 rgba(255,255,255,0.12) inset;
    }
    .bp:hover {
      box-shadow: 0 6px 28px rgba(124,111,255,0.5), 0 1px 0 rgba(255,255,255,0.15) inset;
      transform: translateY(-2px) scale(1.02);
      filter: brightness(1.1);
    }
    .bp:disabled { opacity:.45; cursor:not-allowed; transform:none; }

    .bs {
      background: rgba(124,111,255,0.06);
      color: ${T.sub};
      border: 1px solid ${T.border};
    }
    .bs:hover {
      background: rgba(124,111,255,0.12);
      border-color: ${T.borderMed};
      color: ${T.text};
      transform: translateY(-1px);
    }

    .bg {
      background:transparent; color:${T.sub}; border:none; padding:6px 10px;
    }
    .bg:hover { background:rgba(124,111,255,0.07); color:${T.text}; border-radius:8px; }

    .bd {
      background:transparent; color:${T.red}; border:1px solid rgba(255,91,107,0.2);
    }
    .bd:hover { background:${T.redDim}; transform:translateY(-1px); }

    button:disabled { opacity:.45; cursor:not-allowed; }

    .inp {
      background: rgba(10,10,32,0.8);
      border: 1px solid ${T.border};
      border-radius: 10px;
      padding: 9px 13px;
      font-size: 13px;
      color: ${T.text};
      width: 100%;
      outline: none;
      transition: border-color .18s, box-shadow .18s, background .18s;
    }
    .inp::placeholder { color: ${T.muted}; }
    .inp:focus {
      border-color: ${T.accent};
      box-shadow: 0 0 0 3px rgba(124,111,255,0.18);
      background: rgba(12,12,34,0.95);
    }

    .prose p { margin-bottom:9px; font-size:13px; color:${T.sub}; line-height:1.8; }
    .prose strong { color:${T.text}; font-weight:700; }
    .prose ul, .prose ol { padding-left:18px; margin-bottom:9px; }
    .prose li { margin-bottom:5px; font-size:13px; color:${T.sub}; }
    .prose h3, .prose h4 { color:${T.text}; font-weight:700; margin:12px 0 5px; font-size:13px; }
    .prose h2 { color:${T.text}; font-weight:800; margin:14px 0 6px; font-size:14px; background: linear-gradient(90deg,${T.accent},${T.purple}); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }

    .shimmer-text {
      background: linear-gradient(90deg, ${T.accent} 0%, ${T.purple} 30%, ${T.accent2} 60%, ${T.accent} 100%);
      background-size: 200% auto;
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      animation: shimmer 3s linear infinite;
    }

    .grad-border {
      position: relative;
    }
    .grad-border::before {
      content: '';
      position: absolute;
      inset: -1px;
      border-radius: 15px;
      background: linear-gradient(135deg, ${T.accent}, ${T.purple}, ${T.accent2});
      z-index: -1;
      opacity: 0.6;
    }

    .sidebar-nav-item {
      display: flex; align-items: center; gap: 10px;
      width: 100%; border-radius: 10px; margin-bottom: 3px;
      padding: 10px 12px; cursor: pointer;
      border: 1px solid transparent;
      font-size: 13px; font-weight: 500;
      transition: all .2s cubic-bezier(.22,1,.36,1);
      position: relative; overflow: hidden;
    }
    .sidebar-nav-item.active {
      background: linear-gradient(135deg, rgba(124,111,255,0.18), rgba(191,95,255,0.1));
      border-color: rgba(124,111,255,0.3);
      color: #fff;
      box-shadow: 0 2px 12px rgba(124,111,255,0.2), inset 0 1px 0 rgba(255,255,255,0.06);
    }
    .sidebar-nav-item:not(.active):hover {
      background: rgba(124,111,255,0.07);
      border-color: rgba(124,111,255,0.15);
      color: ${T.text};
      transform: translateX(3px);
    }
    .sidebar-nav-item.active::before {
      content: '';
      position: absolute;
      left: 0; top: 20%; bottom: 20%;
      width: 3px;
      border-radius: 2px;
      background: linear-gradient(${T.accent}, ${T.purple});
    }

    .stat-card {
      background: ${T.card};
      border: 1px solid ${T.border};
      border-radius: 16px;
      padding: 18px 20px;
      display: flex; gap: 14px; align-items: center;
      transition: all .25s cubic-bezier(.22,1,.36,1);
      cursor: default;
      position: relative;
      overflow: hidden;
    }
    .stat-card::after {
      content: '';
      position: absolute;
      top: 0; right: 0;
      width: 80px; height: 80px;
      border-radius: 50%;
      opacity: 0.04;
      transform: translate(25%, -25%);
      transition: opacity .3s;
    }
    .stat-card:hover {
      transform: translateY(-4px) scale(1.01);
      border-color: rgba(124,111,255,0.25);
      box-shadow: 0 12px 36px rgba(0,0,0,0.4);
    }
    .stat-card:hover::after { opacity:0.08; }

    .quick-btn {
      padding: 11px 14px;
      border-radius: 10px;
      border: 1px solid ${T.border};
      background: rgba(124,111,255,0.04);
      cursor: pointer;
      text-align: left;
      font-size: 12px;
      font-weight: 500;
      color: ${T.sub};
      transition: all .2s cubic-bezier(.22,1,.36,1);
    }
    .quick-btn:hover {
      border-color: rgba(124,111,255,0.35);
      background: rgba(124,111,255,0.1);
      color: ${T.text};
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(124,111,255,0.15);
    }

    .chip {
      font-size: 11px; padding: 4px 12px; border-radius: 20px;
      border: 1px solid ${T.border};
      background: transparent; color: ${T.muted};
      white-space: nowrap; flex-shrink: 0; cursor: pointer;
      transition: all .18s;
    }
    .chip:hover {
      border-color: ${T.accent};
      color: ${T.accent};
      background: rgba(124,111,255,0.08);
      transform: translateY(-1px);
    }
  `}</style>
);

// ─── ICONS ────────────────────────────────────────────────────
const Spin = ({ s = 16 }) => (
  <svg width={s} height={s} viewBox="0 0 24 24" fill="none" className="spin">
    <circle cx="12" cy="12" r="9" stroke="rgba(255,255,255,0.1)" strokeWidth="2.5" />
    <path d="M12 3a9 9 0 019 9" stroke={T.accent} strokeWidth="2.5" strokeLinecap="round" />
  </svg>
);
function Ico({ d, s = 16, c = "currentColor", sw = 1.9 }) {
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke={c} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round">
      {Array.isArray(d) ? d.map((p, i) => <path key={i} d={p} />) : <path d={d} />}
    </svg>
  );
}
const P = {
  home:      "M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z",
  users:     ["M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2","M23 21v-2a4 4 0 00-3-3.87","M16 3.13a4 4 0 010 7.75"],
  chat:      "M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z",
  brain:     ["M9.5 2A2.5 2.5 0 017 4.5v0A2.5 2.5 0 009.5 7h5A2.5 2.5 0 0017 4.5v0A2.5 2.5 0 0014.5 2h-5z","M12 7v10","M9 12H7a2 2 0 00-2 2v5","M15 12h2a2 2 0 012 2v5"],
  zap:       "M13 2L3 14h9l-1 8 10-12h-9z",
  plus:      ["M12 5v14","M5 12h14"],
  send:      "M22 2L11 13M22 2L15 22l-4-9-9-4z",
  edit:      ["M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7","M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4z"],
  trash:     ["M3 6h18","M8 6V4h8v2","M19 6l-1 14H6L5 6"],
  check:     "M20 6L9 17l-5-5",
  x:         ["M18 6L6 18","M6 6l12 12"],
  clock:     ["M12 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10-4.477 10-10 10z","M12 6v6l4 2"],
  mail:      ["M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z","M22 6l-10 7L2 6"],
  phone:     "M22 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07A19.5 19.5 0 013.07 8.8 19.79 19.79 0 01.22 4.18 2 2 0 012.2 2h3a2 2 0 012 1.72c.127.96.361 1.903.7 2.81a2 2 0 01-.45 2.11L6.91 9.91a16 16 0 006.17 6.17l1.48-1.48a2 2 0 012.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0122 16.92z",
  bldg:      ["M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z","M9 22V12h6v10"],
  rupee:     ["M6 3h12","M6 8h12","M9 8v13","M9 13l8 9"],
  eye:       ["M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z","M12 15a3 3 0 100-6 3 3 0 000 6z"],
  eyeOff:    ["M17.94 17.94A10.07 10.07 0 0112 20c-7 0-11-8-11-8a18.45 18.45 0 015.06-5.94","M9.9 4.24A9.12 9.12 0 0112 4c7 0 11 8 11 8a18.5 18.5 0 01-2.16 3.19","M1 1l22 22","M14.12 14.12a3 3 0 01-4.24-4.24"],
  award:     ["M12 15a7 7 0 100-14 7 7 0 000 14z","M8.21 13.89L7 23l5-3 5 3-1.21-9.12"],
  copy:      ["M8 4v12a2 2 0 002 2h8a2 2 0 002-2V7.242a2 2 0 00-.602-1.43L16.083 2.57A2 2 0 0014.685 2H10a2 2 0 00-2 2z","M16 18v2a2 2 0 01-2 2H6a2 2 0 01-2-2V9a2 2 0 012-2h2"],
  search:    ["M21 21l-4.35-4.35","M17 11A6 6 0 105 11a6 6 0 0012 0z"],
  info:      ["M12 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10-4.477 10-10 10z","M12 16v-4","M12 8h.01"],
  target:    ["M12 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10-4.477 10-10 10z","M12 18a6 6 0 100-12 6 6 0 000 12z","M12 14a2 2 0 100-4 2 2 0 000 4z"],
  briefcase: ["M20 7H4a2 2 0 00-2 2v10a2 2 0 002 2h16a2 2 0 002-2V9a2 2 0 00-2-2z","M16 7V5a2 2 0 00-2-2h-4a2 2 0 00-2 2v2"],
  logout:    ["M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4","M16 17l5-5-5-5","M21 12H9"],
};

// ─── SHARED COMPONENTS ────────────────────────────────────────
function Av({ name, size = 36 }) {
  const l = (name || "?").split(" ").map(w => w[0]).slice(0, 2).join("").toUpperCase();
  const hue = ((name || "").charCodeAt(0) * 37) % 360;
  return (
    <div style={{
      width: size, height: size, borderRadius: "50%", flexShrink: 0,
      background: `linear-gradient(135deg, hsl(${hue},60%,25%), hsl(${hue+40},55%,18%))`,
      border: `1.5px solid hsl(${hue},60%,35%)`,
      color: `hsl(${hue},80%,75%)`,
      display: "flex", alignItems: "center", justifyContent: "center",
      fontSize: size * .33, fontWeight: 700,
      boxShadow: `0 2px 10px hsl(${hue},60%,25%)44`
    }}>{l}</div>
  );
}
function Tag({ children, color = T.accent, bg }) {
  return (
    <span className="tag" style={{
      color,
      background: bg || `${color}18`,
      border: `1px solid ${color}28`,
    }}>{children}</span>
  );
}
function PriTag({ p }) { const m = { high: [T.red, "High"], medium: [T.amber, "Med"], low: [T.sub, "Low"] }; const [c, l] = m[p] || m.low; return <Tag color={c}>{l}</Tag>; }
function StTag({ s }) { const m = { discovery: [T.blue, "Discovery"], negotiating: [T.amber, "Negotiating"], proposal: [T.accent, "Proposal"], won: [T.green, "Won 🎉"], lost: [T.red, "Lost"] }; const [c, l] = m[s] || [T.sub, s]; return <Tag color={c}>{l}</Tag>; }
function MdTag({ tier }) { return tier === "Pro" ? <Tag color={T.purple}>⚡ Pro</Tag> : <Tag color={T.green}>💨 Flash</Tag>; }
function Prose({ text }) {
  if (!text) return null;
  return (
    <div className="prose">
      {text.split("\n").map((line, i) => {
        if (!line.trim()) return <br key={i} />;
        const parts = line.split(/(\*\*[^*]+\*\*)/g).map((p, j) =>
          p.startsWith("**") && p.endsWith("**") ? <strong key={j}>{p.slice(2, -2)}</strong> : p
        );
        if (/^#{1,4} /.test(line)) return <h3 key={i}>{parts}</h3>;
        if (line.startsWith("- ") || line.startsWith("• ")) return <li key={i}>{parts.slice(1)}</li>;
        if (/^\d+\. /.test(line)) return <li key={i}>{line.replace(/^\d+\. /, "")}</li>;
        return <p key={i}>{parts}</p>;
      })}
    </div>
  );
}
function Bar({ val, max = 100, color = T.accent }) {
  return (
    <div style={{ height: 6, background: "rgba(255,255,255,0.05)", borderRadius: 4 }}>
      <div style={{
        width: `${Math.min(100, (val / max) * 100)}%`, height: "100%",
        background: `linear-gradient(90deg, ${color}, ${color}99)`,
        borderRadius: 4, transition: "width .6s cubic-bezier(.22,1,.36,1)",
        boxShadow: `0 0 8px ${color}55`
      }} />
    </div>
  );
}

// ─── SPLASH / LOADING SCREEN ──────────────────────────────────
function SplashScreen({ onDone }) {
  const [phase, setPhase] = useState(0);
  const [progress, setProgress] = useState(0);
  const steps = ["Initialising NegotiAI…", "Loading Hindsight Memory…", "Connecting CascadeFlow…", "Ready to negotiate!"];

  useEffect(() => {
    const t1 = setInterval(() => setProgress(p => Math.min(p + Math.random() * 12, 95)), 120);
    const phases = [600, 1300, 2000, 2700];
    const timers = phases.map((d, i) => setTimeout(() => setPhase(i), d));
    setTimeout(() => {
      setProgress(100);
      clearInterval(t1);
      setTimeout(onDone, 500);
    }, 3200);
    return () => { clearInterval(t1); timers.forEach(clearTimeout); };
  }, []);

  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 9999,
      background: T.bg,
      display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
      overflow: "hidden",
    }}>
      {/* Animated bg orbs */}
      {[[T.accent,"30%","20%","500px"],[T.purple,"70%","60%","400px"],[T.accent2,"20%","70%","350px"]].map(([c,x,y,s],i)=>(
        <div key={i} style={{
          position:"absolute", left:x, top:y, width:s, height:s,
          borderRadius:"50%", background:c,
          filter:"blur(140px)", opacity:0.12,
          animation:`bgPulse ${2+i*0.8}s ease-in-out infinite`,
          animationDelay:`${i*0.5}s`,
          transform:"translate(-50%,-50%)"
        }}/>
      ))}
      {/* Orbit rings */}
      <div style={{ position:"absolute", width:500, height:500, borderRadius:"50%", border:`1px solid rgba(124,111,255,0.08)` }} />
      <div style={{ position:"absolute", width:380, height:380, borderRadius:"50%", border:`1px solid rgba(124,111,255,0.12)`, animation:"orbitSpin 20s linear infinite" }}>
        <div style={{ position:"absolute", top:-5, left:"50%", width:10, height:10, borderRadius:"50%", background: T.accent, boxShadow:`0 0 15px ${T.accent}`, transform:"translateX(-50%)" }}/>
      </div>
      <div style={{ position:"absolute", width:260, height:260, borderRadius:"50%", border:`1px solid rgba(191,95,255,0.1)`, animation:"orbitSpin 14s linear infinite reverse" }}>
        <div style={{ position:"absolute", bottom:-5, left:"50%", width:8, height:8, borderRadius:"50%", background: T.purple, boxShadow:`0 0 12px ${T.purple}`, transform:"translateX(-50%)" }}/>
      </div>

      {/* Central logo */}
      <div className="float" style={{ display:"flex", flexDirection:"column", alignItems:"center", zIndex:10 }}>
        <div className="pulse-glow" style={{
          width:88, height:88, borderRadius:24,
          background:`linear-gradient(135deg,${T.accent},${T.purple})`,
          display:"flex", alignItems:"center", justifyContent:"center",
          marginBottom:28,
          boxShadow:`0 0 40px rgba(124,111,255,0.5), 0 0 80px rgba(124,111,255,0.2)`,
        }}>
          <Ico d={P.zap} s={42} c="#fff" sw={2.5} />
        </div>

        <div style={{ fontSize:42, fontWeight:900, fontFamily:"'Space Grotesk',sans-serif", letterSpacing:"-1px", marginBottom:8 }}
          className="shimmer-text">NegotiAI</div>
        <div style={{ fontSize:14, color:T.sub, marginBottom:2 }}>AI-Powered Sales Negotiation Platform</div>
        <div style={{ fontSize:12, color:T.muted, marginBottom:40 }}>Built for India 🇮🇳</div>

        {/* Progress bar */}
        <div style={{ width:280 }}>
          <div style={{ display:"flex", justifyContent:"space-between", marginBottom:8 }}>
            <span style={{ fontSize:12, color:T.sub }}>{steps[Math.min(phase, steps.length-1)]}</span>
            <span style={{ fontSize:12, color:T.accent, fontWeight:700 }}>{Math.round(progress)}%</span>
          </div>
          <div style={{ height:3, background:"rgba(255,255,255,0.06)", borderRadius:4, overflow:"hidden" }}>
            <div style={{
              height:"100%", borderRadius:4,
              background:`linear-gradient(90deg,${T.accent},${T.purple},${T.accent2})`,
              backgroundSize:"200% 100%",
              width:`${progress}%`,
              transition:"width .15s ease",
              animation:"shimmer 2s linear infinite",
            }}/>
          </div>
        </div>

        {/* Feature dots */}
        <div style={{ display:"flex", gap:24, marginTop:32 }}>
          {["Hindsight Memory","CascadeFlow AI","Gemini 1.5"].map((f,i)=>(
            <div key={f} style={{ display:"flex", alignItems:"center", gap:6, fontSize:11, color:T.muted, opacity: phase >= i ? 1 : 0.3, transition:"opacity .4s" }}>
              <div style={{ width:6, height:6, borderRadius:"50%", background:T.green, boxShadow:`0 0 8px ${T.green}` }} /> {f}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// LOGIN PAGE
// ═══════════════════════════════════════════════════════════════
function LoginPage({ onLogin }) {
  const [userId,   setUserId]   = useState("");
  const [password, setPassword] = useState("");
  const [error,    setError]    = useState("");
  const [loading,  setLoading]  = useState(false);
  const [showPw,   setShowPw]   = useState(false);

  function handleSubmit(e) {
    if (e && e.preventDefault) e.preventDefault();
    setError("");
    if (!userId.trim() || !password) { setError("Please enter your User ID and Password."); return; }
    setLoading(true);
    setTimeout(() => {
      if (userId.trim() === VALID_USER.id && password === VALID_USER.pass) {
        onLogin({ userId: VALID_USER.id, name: VALID_USER.name, role: VALID_USER.role });
      } else {
        setError("Invalid User ID or Password. Please check and try again.");
        setLoading(false);
      }
    }, 700);
  }

  return (
    <div style={{ minHeight:"100vh", background:T.bg, display:"flex", alignItems:"center", justifyContent:"center", padding:20, position:"relative", overflow:"hidden" }}>
      {/* Background orbs */}
      {[[T.accent,"25%","40%"],[T.purple,"78%","65%"],[T.accent2,"60%","15%"]].map(([c,x,y],i)=>(
        <div key={i} style={{ position:"absolute", left:x, top:y, width:"500px", height:"500px", borderRadius:"50%", background:c, filter:"blur(160px)", opacity:0.07, transform:"translate(-50%,-50%)", animation:`bgPulse ${3+i}s ease-in-out infinite` }}/>
      ))}
      {/* Grid lines */}
      <div style={{ position:"absolute", inset:0, backgroundImage:`linear-gradient(rgba(124,111,255,0.04) 1px,transparent 1px),linear-gradient(90deg,rgba(124,111,255,0.04) 1px,transparent 1px)`, backgroundSize:"48px 48px", pointerEvents:"none" }}/>

      <div style={{ width:"100%", maxWidth:440, position:"relative" }} className="asi">
        {/* Logo */}
        <div style={{ textAlign:"center", marginBottom:36 }}>
          <div className="float" style={{ display:"inline-flex", alignItems:"center", gap:14, marginBottom:16 }}>
            <div className="pulse-glow" style={{
              width:56, height:56, borderRadius:16,
              background:`linear-gradient(135deg,${T.accent},${T.purple})`,
              display:"flex", alignItems:"center", justifyContent:"center",
            }}>
              <Ico d={P.zap} s={28} c="#fff" sw={2.3} />
            </div>
            <span style={{ fontSize:32, fontWeight:900, fontFamily:"'Space Grotesk',sans-serif", letterSpacing:"-0.8px" }} className="shimmer-text">NegotiAI</span>
          </div>
          <div style={{ fontSize:13, color:T.sub }}>AI-Powered Sales Negotiation Platform</div>
          <div style={{ fontSize:12, color:T.muted, marginTop:3 }}>Built for India 🇮🇳</div>
        </div>

        {/* Form card */}
        <div style={{
          background: T.card,
          border: `1px solid ${T.borderMed}`,
          borderRadius:18,
          padding:32,
          boxShadow:`0 24px 80px rgba(0,0,0,0.5), 0 0 0 1px rgba(124,111,255,0.1) inset`,
        }}>
          <h2 style={{ fontSize:18, fontWeight:800, marginBottom:6 }}>Sign in to your account</h2>
          <p style={{ fontSize:13, color:T.muted, marginBottom:26 }}>Enter your credentials to continue</p>

          {error && (
            <div style={{
              background:T.redDim, border:`1px solid rgba(255,91,107,0.3)`,
              borderRadius:10, padding:"11px 14px", marginBottom:18,
              fontSize:13, color:T.red, display:"flex", gap:8, alignItems:"flex-start",
              animation:"scaleIn .2s ease"
            }}>
              <Ico d={P.info} s={15} c={T.red} /> <span>{error}</span>
            </div>
          )}

          <div>
            <div style={{ marginBottom:16 }}>
              <label style={{ display:"block", fontSize:11, color:T.sub, marginBottom:6, fontWeight:700, letterSpacing:".06em" }}>USER ID</label>
              <input className="inp" type="text" placeholder="Enter your User ID" value={userId}
                onChange={e => setUserId(e.target.value)}
                onKeyDown={e => e.key === "Enter" && handleSubmit(e)}
                disabled={loading} autoComplete="username" />
            </div>
            <div style={{ marginBottom:26 }}>
              <label style={{ display:"block", fontSize:11, color:T.sub, marginBottom:6, fontWeight:700, letterSpacing:".06em" }}>PASSWORD</label>
              <div style={{ position:"relative" }}>
                <input className="inp" type={showPw?"text":"password"} placeholder="Enter your password"
                  value={password} onChange={e => setPassword(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && handleSubmit(e)}
                  disabled={loading}
                  autoComplete="current-password" style={{ paddingRight:42 }} />
                <button type="button" onClick={() => setShowPw(v=>!v)}
                  style={{ position:"absolute", right:11, top:"50%", transform:"translateY(-50%)", background:"none", border:"none", cursor:"pointer", color:T.muted, padding:2, display:"flex" }}>
                  <Ico d={showPw ? P.eyeOff : P.eye} s={15} c={T.muted} />
                </button>
              </div>
            </div>
            <button className="btn bp"
              style={{ width:"100%", justifyContent:"center", padding:"13px", fontSize:14, fontWeight:700 }}
              onClick={handleSubmit}
              disabled={loading}>
              {loading ? <><Spin s={16} /> Signing in…</> : "Sign In →"}
            </button>
          </div>

          <div style={{ marginTop:22, padding:"13px 16px", background:T.accentDim, borderRadius:12, border:`1px solid ${T.accentBd}` }}>
            <div style={{ fontSize:11, fontWeight:700, color:T.sub, marginBottom:8, letterSpacing:".06em" }}>DEMO CREDENTIALS</div>
            <div style={{ display:"flex", justifyContent:"space-between", fontSize:12 }}>
              <span style={{ color:T.muted }}>User ID: <span style={{ color:T.accent, fontWeight:700 }}>test_01</span></span>
              <span style={{ color:T.muted }}>Password: <span style={{ color:T.accent, fontWeight:700 }}>test@1234</span></span>
            </div>
          </div>
        </div>

        <div style={{ display:"flex", justifyContent:"center", gap:24, marginTop:24 }}>
          {["Hindsight Memory","CascadeFlow AI","Gemini 1.5"].map(f => (
            <div key={f} style={{ display:"flex", alignItems:"center", gap:5, fontSize:11, color:T.muted }}>
              <div style={{ width:5, height:5, borderRadius:"50%", background:T.green, boxShadow:`0 0 6px ${T.green}` }} /> {f}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// SIDEBAR
// ═══════════════════════════════════════════════════════════════
const NAV = [
  { id:"dashboard", label:"Dashboard",      icon:P.home  },
  { id:"customers", label:"Customers",       icon:P.users },
  { id:"chat",      label:"AI Negotiation",  icon:P.chat  },
  { id:"memory",    label:"Memory Timeline", icon:P.brain },
  { id:"runtime",   label:"Runtime",         icon:P.zap   },
];
function Sidebar({ page, nav, user, onLogout }) {
  return (
    <aside style={{
      width:220, minHeight:"100vh",
      background:`linear-gradient(180deg, ${T.surface} 0%, rgba(7,7,26,0.97) 100%)`,
      borderRight:`1px solid ${T.border}`,
      display:"flex", flexDirection:"column", padding:"18px 12px",
      position:"fixed", left:0, top:0, bottom:0, zIndex:200,
      boxShadow:`4px 0 24px rgba(0,0,0,0.3)`,
    }}>
      {/* Logo */}
      <div style={{ display:"flex", alignItems:"center", gap:10, padding:"6px 8px", marginBottom:28 }}>
        <div style={{
          width:36, height:36, borderRadius:10,
          background:`linear-gradient(135deg,${T.accent},${T.purple})`,
          display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0,
          boxShadow:`0 4px 16px rgba(124,111,255,0.4)`,
        }}>
          <Ico d={P.zap} s={18} c="#fff" sw={2.4} />
        </div>
        <div>
          <div style={{ fontSize:16, fontWeight:900, letterSpacing:"-0.5px", fontFamily:"'Space Grotesk',sans-serif" }} className="shimmer-text">NegotiAI</div>
          <div style={{ fontSize:10, color:T.muted }}>India 🇮🇳</div>
        </div>
      </div>

      {/* Nav */}
      <nav style={{ flex:1 }}>
        <div style={{ fontSize:10, color:T.muted, fontWeight:700, letterSpacing:".1em", padding:"0 8px", marginBottom:10 }}>NAVIGATION</div>
        {NAV.map((n,i) => {
          const a = page === n.id;
          return (
            <button key={n.id} className={`sidebar-nav-item${a?" active":""}`}
              onClick={() => nav(n.id)}
              style={{ animationDelay:`${i*0.05}s`, fontFamily:"inherit", background:"none", border:"none", cursor:"pointer", color: a ? "#fff" : T.sub }}
            >
              <Ico d={n.icon} s={15} c={a ? T.accent : T.muted} sw={a?2.2:1.9} />
              {n.label}
              {a && <div style={{ marginLeft:"auto", width:6, height:6, borderRadius:"50%", background:T.accent, boxShadow:`0 0 8px ${T.accent}` }}/>}
            </button>
          );
        })}
      </nav>

      {/* User */}
      <div style={{ borderTop:`1px solid ${T.border}`, paddingTop:16 }}>
        <div style={{ display:"flex", alignItems:"center", gap:10, padding:"0 6px", marginBottom:12 }}>
          <Av name={user?.name} size={34} />
          <div style={{ minWidth:0 }}>
            <div style={{ fontSize:12, fontWeight:700, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{user?.name}</div>
            <div style={{ fontSize:10, color:T.muted }}>{user?.userId}</div>
          </div>
        </div>
        <button className="btn bs" style={{ width:"100%", justifyContent:"center", fontSize:12, padding:"8px" }} onClick={onLogout}>
          <Ico d={P.logout} s={13} /> Sign Out
        </button>
      </div>
    </aside>
  );
}

// ═══════════════════════════════════════════════════════════════
// DASHBOARD
// ═══════════════════════════════════════════════════════════════
function Dashboard({ customers, auditLogs, nav, setAC }) {
  const totalMem  = customers.reduce((s,c) => s + HS.load(c.id).length, 0);
  const totalCost = auditLogs.reduce((s,l) => s + (l.costINR||0), 0);
  const flashSave = auditLogs.filter(l=>l.tier==="Flash").reduce((s,l) => s + (l.totalTokens||0)*(0.00375-0.000075)/1000*83, 0);
  const won       = customers.filter(c=>c.status==="won").length;
  const active    = customers.filter(c=>!["won","lost"].includes(c.status)).length;
  const stats = [
    { l:"Total Customers",     v:customers.length,           icon:P.users,  c:T.accent },
    { l:"Active Deals",        v:active,                     icon:P.chat,   c:T.blue   },
    { l:"Hindsight Memories",  v:totalMem,                   icon:P.brain,  c:T.purple },
    { l:"Deals Won",           v:won,                        icon:P.award,  c:T.green  },
    { l:"AI Spend",            v:`₹${totalCost.toFixed(2)}`, icon:P.rupee,  c:T.amber  },
    { l:"CascadeFlow Savings", v:`₹${flashSave.toFixed(2)}`, icon:P.zap,    c:T.green  },
  ];
  return (
    <div className="au" style={{ maxWidth:1100 }}>
      <div style={{ marginBottom:28 }}>
        <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:6 }}>
          <h1 style={{ fontSize:22, fontWeight:900, letterSpacing:"-0.5px", fontFamily:"'Space Grotesk',sans-serif" }}
            className="shimmer-text">Dashboard</h1>
          <Tag color={T.green}>Live</Tag>
        </div>
        <p style={{ color:T.sub, fontSize:13 }}>Real-time overview of your AI negotiation pipeline</p>
      </div>

      <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:14, marginBottom:24 }}>
        {stats.map((s,i) => (
          <div key={s.l} className="stat-card card-3d" style={{ animationDelay:`${i*0.06}s` }}>
            <div style={{ width:44, height:44, borderRadius:12, background:`${s.c}18`, border:`1px solid ${s.c}22`, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0, boxShadow:`0 4px 16px ${s.c}22` }}>
              <Ico d={s.icon} s={20} c={s.c} />
            </div>
            <div>
              <div style={{ fontSize:24, fontWeight:900, lineHeight:1.1, color:s.c, fontFamily:"'Space Grotesk',sans-serif" }}>{s.v}</div>
              <div style={{ fontSize:11, color:T.muted, marginTop:3 }}>{s.l}</div>
            </div>
            <div style={{ position:"absolute", top:0, right:0, width:80, height:80, borderRadius:"50%", background:s.c, filter:"blur(40px)", opacity:0.06, transform:"translate(25%,-25%)" }}/>
          </div>
        ))}
      </div>

      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:18 }}>
        <div className="card" style={{ padding:22 }}>
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:16 }}>
            <span style={{ fontWeight:700, fontSize:14 }}>Customer Pipeline</span>
            <button className="btn bg" style={{ fontSize:12 }} onClick={() => nav("customers")}>View all →</button>
          </div>
          {customers.map(c => {
            const mems = HS.load(c.id);
            return (
              <div key={c.id} onClick={() => { setAC(c); nav("chat"); }}
                style={{ display:"flex", alignItems:"center", gap:12, padding:"10px 0", borderBottom:`1px solid ${T.border}`, cursor:"pointer", transition:"all .15s" }}
                onMouseEnter={e => e.currentTarget.style.paddingLeft = "6px"}
                onMouseLeave={e => e.currentTarget.style.paddingLeft = "0px"}>
                <Av name={c.name} size={36} />
                <div style={{ flex:1, minWidth:0 }}>
                  <div style={{ fontWeight:600, fontSize:13 }}>{c.name}</div>
                  <div style={{ color:T.muted, fontSize:12, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{c.company}</div>
                </div>
                <div style={{ display:"flex", gap:6, flexShrink:0 }}>
                  <StTag s={c.status} />
                  {mems.length > 0 && <Tag color={T.purple}>{mems.length}💾</Tag>}
                </div>
              </div>
            );
          })}
        </div>
        <div className="card" style={{ padding:22 }}>
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:16 }}>
            <span style={{ fontWeight:700, fontSize:14 }}>Recent AI Calls</span>
            <button className="btn bg" style={{ fontSize:12 }} onClick={() => nav("runtime")}>Full log →</button>
          </div>
          {auditLogs.length === 0 && <p style={{ color:T.muted, fontSize:13 }}>No AI calls yet. Start a negotiation!</p>}
          {auditLogs.slice(0, 6).map(l => (
            <div key={l.id} style={{ padding:"9px 0", borderBottom:`1px solid ${T.border}` }}>
              <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:3 }}>
                <span style={{ fontSize:12, fontWeight:600 }}>{l.customer}</span>
                <MdTag tier={l.tier} />
              </div>
              <div style={{ display:"flex", gap:8, fontSize:11, color:T.muted, flexWrap:"wrap" }}>
                <span>{l.latency}ms</span><span>·</span><span>{l.totalTokens} tok</span><span>·</span>
                <span>₹{(l.costINR||0).toFixed(4)}</span><span>·</span>
                <span style={{ textTransform:"capitalize" }}>{(l.taskType||"").replace(/_/g," ")}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// CUSTOMER FORM
// ═══════════════════════════════════════════════════════════════
function CustForm({ init, onSave, onClose }) {
  const blank = { name:"",company:"",email:"",phone:"",city:"",industry:"",budget:"",requirements:"",objections:"",competitors:"",style:"formal",priority:"medium",status:"discovery" };
  const [f, setF] = useState(init || blank);
  const [saving, setSaving] = useState(false);
  const u = k => e => setF(p => ({ ...p, [k]: e.target.value }));
  const save = async () => {
    if (!f.name.trim() || !f.company.trim()) { alert("Name and company required."); return; }
    setSaving(true);
    await new Promise(r => setTimeout(r, 250));
    onSave({ ...f, id: f.id||`c${Date.now()}`, createdAt: f.createdAt||new Date().toISOString() });
    setSaving(false);
  };
  return (
    <div style={{ position:"fixed", inset:0, background:"rgba(0,0,0,0.85)", backdropFilter:"blur(8px)", zIndex:500, display:"flex", alignItems:"center", justifyContent:"center", padding:20 }}
      onClick={e => e.target===e.currentTarget && onClose()}>
      <div className="card asi" style={{ width:"100%", maxWidth:640, maxHeight:"92vh", overflowY:"auto", padding:30, borderColor:T.borderMed, boxShadow:`0 32px 100px rgba(0,0,0,0.6)` }}>
        <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:24 }}>
          <h2 style={{ fontSize:18, fontWeight:800, fontFamily:"'Space Grotesk',sans-serif" }}>{init?"Edit Customer":"Add New Customer"}</h2>
          <button className="btn bg" onClick={onClose}><Ico d={P.x} s={16} /></button>
        </div>
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:14 }}>
          {[["Full Name *","name","Rajesh Mehta"],["Company *","company","TechBridge Pvt Ltd"],["Email","email","raj@company.in"],["Phone","phone","+91-98765-43210"],["City","city","Mumbai"],["Industry","industry","IT / Healthcare / Retail"],["Budget","budget","₹5,00,000 – ₹10,00,000"],["Competitors","competitors","Zoho CRM, Salesforce"]].map(([label,key,ph]) => (
            <div key={key}>
              <label style={{ display:"block", fontSize:11, color:T.sub, marginBottom:6, fontWeight:700, letterSpacing:".05em" }}>{label}</label>
              <input className="inp" placeholder={ph} value={f[key]||""} onChange={u(key)} />
            </div>
          ))}
        </div>
        {[["Requirements","requirements","Describe technical and business requirements…",80],["Known Objections","objections","Price concerns, compliance, timeline worries…",65]].map(([label,key,ph,h]) => (
          <div key={key} style={{ marginTop:14 }}>
            <label style={{ display:"block", fontSize:11, color:T.sub, marginBottom:6, fontWeight:700, letterSpacing:".05em" }}>{label}</label>
            <textarea className="inp" style={{ minHeight:h, resize:"vertical" }} placeholder={ph} value={f[key]||""} onChange={u(key)} />
          </div>
        ))}
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:14, marginTop:14 }}>
          {[["Style","style",[["formal","Formal"],["casual","Casual"],["technical","Technical"],["concise","Concise"]]],["Priority","priority",[["high","High"],["medium","Medium"],["low","Low"]]],["Status","status",[["discovery","Discovery"],["negotiating","Negotiating"],["proposal","Proposal"],["won","Won"],["lost","Lost"]]]].map(([label,key,opts]) => (
            <div key={key}>
              <label style={{ display:"block", fontSize:11, color:T.sub, marginBottom:6, fontWeight:700, letterSpacing:".05em" }}>{label}</label>
              <select className="inp" value={f[key]} onChange={u(key)}>{opts.map(([v,l]) => <option key={v} value={v}>{l}</option>)}</select>
            </div>
          ))}
        </div>
        <div style={{ display:"flex", gap:10, marginTop:24 }}>
          <button className="btn bp" onClick={save} disabled={saving} style={{ flex:1, justifyContent:"center" }}>
            {saving ? <><Spin s={14} /> Saving…</> : <><Ico d={P.check} s={14} c="#fff" />{init?"Update Customer":"Add Customer"}</>}
          </button>
          <button className="btn bs" onClick={onClose}>Cancel</button>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// CUSTOMERS PAGE
// ═══════════════════════════════════════════════════════════════
function Customers({ customers, setCustomers, nav, setAC }) {
  const [form,   setForm]   = useState(null);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("all");
  const filtered = customers.filter(c => {
    const q = search.toLowerCase();
    return (!q || c.name.toLowerCase().includes(q) || c.company.toLowerCase().includes(q) || (c.industry||"").toLowerCase().includes(q))
        && (filter==="all" || c.status===filter || c.priority===filter);
  });
  const save = d => { setCustomers(p => { const e = p.find(c=>c.id===d.id); return e?p.map(c=>c.id===d.id?d:c):[...p,d]; }); setForm(null); };
  const del  = id => { if (!confirm("Delete this customer and all their memories?")) return; localStorage.removeItem(`hs_${id}`); localStorage.removeItem(`nai_chat_${id}`); setCustomers(p=>p.filter(c=>c.id!==id)); };
  return (
    <div className="au" style={{ maxWidth:1100 }}>
      {form && <CustForm init={form==="new"?null:form} onSave={save} onClose={() => setForm(null)} />}
      <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:26 }}>
        <div>
          <h1 style={{ fontSize:22, fontWeight:900, letterSpacing:"-0.5px", fontFamily:"'Space Grotesk',sans-serif", marginBottom:4 }}
            className="shimmer-text">Customer Management</h1>
          <p style={{ color:T.sub, fontSize:13 }}>Manage your pipeline. Every customer is backed by Hindsight memory.</p>
        </div>
        <button className="btn bp" onClick={() => setForm("new")}><Ico d={P.plus} s={15} c="#fff" /> Add Customer</button>
      </div>
      <div style={{ display:"flex", gap:10, marginBottom:20 }}>
        <div style={{ position:"relative", flex:1 }}>
          <div style={{ position:"absolute", left:12, top:"50%", transform:"translateY(-50%)" }}><Ico d={P.search} s={14} c={T.muted} /></div>
          <input className="inp" style={{ paddingLeft:36 }} placeholder="Search name, company, industry…" value={search} onChange={e => setSearch(e.target.value)} />
        </div>
        <select className="inp" style={{ width:170 }} value={filter} onChange={e => setFilter(e.target.value)}>
          <option value="all">All Statuses</option>
          <option value="discovery">Discovery</option>
          <option value="negotiating">Negotiating</option>
          <option value="proposal">Proposal</option>
          <option value="won">Won</option>
          <option value="lost">Lost</option>
          <option value="high">High Priority</option>
        </select>
      </div>
      {filtered.length===0 && <div className="card" style={{ padding:60, textAlign:"center", color:T.muted }}><Ico d={P.users} s={38} c={T.muted} /><p style={{ marginTop:14, fontSize:14, color:T.sub }}>No customers found</p></div>}
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16 }}>
        {filtered.map((c,i) => {
          const mems = HS.load(c.id);
          return (
            <div key={c.id} className="card card-3d" style={{ padding:22, animationDelay:`${i*0.05}s` }}>
              <div style={{ display:"flex", gap:13, marginBottom:16 }}>
                <Av name={c.name} size={44} />
                <div style={{ flex:1, minWidth:0 }}>
                  <div style={{ display:"flex", alignItems:"center", gap:7, marginBottom:4, flexWrap:"wrap" }}>
                    <span style={{ fontWeight:700, fontSize:14 }}>{c.name}</span>
                    <PriTag p={c.priority} /><StTag s={c.status} />
                  </div>
                  <div style={{ fontSize:12, color:T.muted }}>{c.company} · {c.city}</div>
                  <div style={{ fontSize:12, color:T.muted }}>{c.industry}</div>
                </div>
                <div style={{ display:"flex", gap:5, flexShrink:0 }}>
                  <button className="btn bg" style={{ padding:"6px 8px" }} onClick={() => setForm(c)}><Ico d={P.edit} s={13} /></button>
                  <button className="btn bd"  style={{ padding:"6px 8px" }} onClick={() => del(c.id)}><Ico d={P.trash} s={13} /></button>
                </div>
              </div>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:9, marginBottom:16 }}>
                {[[P.rupee,c.budget||"Budget TBD"],[P.brain,`${mems.length} memories`],[P.users,c.style||"formal"],[P.phone,c.phone||"—"]].map(([ico,val],i) => (
                  <div key={i} style={{ display:"flex", alignItems:"center", gap:6, fontSize:12, color:T.muted }}>
                    <Ico d={ico} s={12} c={T.muted} /><span style={{ overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{val}</span>
                  </div>
                ))}
              </div>
              {c.requirements && <div style={{ fontSize:12, color:T.sub, background:"rgba(124,111,255,0.05)", borderRadius:8, padding:"9px 12px", marginBottom:16, lineHeight:1.7, borderLeft:`2px solid ${T.accent}` }}>{c.requirements.slice(0,110)}{c.requirements.length>110?"…":""}</div>}
              <button className="btn bp" style={{ width:"100%", justifyContent:"center" }} onClick={() => { setAC(c); nav("chat"); }}>
                <Ico d={P.chat} s={14} c="#fff" /> Start AI Negotiation
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// CHAT PAGE
// ═══════════════════════════════════════════════════════════════
function Chat({ customers, setCustomers, ac, setAC, setAuditLogs, nav }) {
  const [msgs,      setMsgs]      = useDB(`chat_${ac?.id}`, []);
  const [input,     setInput]     = useState("");
  const [streaming, setStreaming] = useState(false);
  const [streamTxt, setStreamTxt] = useState("");
  const [routeInfo, setRouteInfo] = useState(null);
  const [copied,    setCopied]    = useState(null);
  const bottomRef = useRef(null);
  const inputRef  = useRef(null);

  useEffect(() => { if (ac) { const s = dbGet(`chat_${ac.id}`)||[]; setMsgs(s); } }, [ac?.id]);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior:"smooth" }); }, [msgs, streamTxt]);

  const customer = customers.find(c=>c.id===ac?.id) || ac;
  const mems     = customer ? HS.load(customer.id) : [];

  const send = async (text) => {
    const msg = (text||input).trim();
    if (!msg||streaming||!customer) return;
    setInput("");
    const um = { id:`u${Date.now()}`, role:"user", content:msg, ts:new Date().toISOString() };
    const newMsgs = [...msgs, um];
    setMsgs(newMsgs);
    const route    = cascadeRoute(msg);
    const taskType = detectTask(msg);
    setRouteInfo(route); setStreaming(true); setStreamTxt("");
    try {
      const sp      = buildSP(customer);
      const history = newMsgs.slice(-13,-1).map(m=>({ role:m.role, content:m.content }));
      const result  = await callGemini({ model:route.model, systemPrompt:sp, history, userMessage:msg, onChunk:(f)=>setStreamTxt(f) });
      const am = { id:`a${Date.now()}`, role:"assistant", content:result.text, ts:new Date().toISOString(), meta:{ model:result.model, tier:route.tier, reason:route.reason, latency:result.latency, tokens:result.totalTokens, inputTokens:result.inputTokens, outputTokens:result.outputTokens, costINR:result.costINR, taskType } };
      setMsgs([...newMsgs, am]);
      if (taskType!=="conversation"||msg.length>100||result.text.length>150) {
        const mt = taskType==="follow_up_email"?"followup":taskType==="proposal"?"proposal":taskType==="meeting_summary"?"meeting":taskType==="objection_handling"?"objection":"note";
        HS.add(customer.id, { id:`m${Date.now()}`, type:mt, content:`${msg.slice(0,100)} → ${result.text.slice(0,200)}`, date:new Date().toISOString(), sentiment:"positive" });
        setCustomers(p=>[...p]);
      }
      setAuditLogs(p=>[{ id:`l${Date.now()}`, ts:new Date().toISOString(), customer:customer.name, customerId:customer.id, taskType, model:result.model, tier:route.tier, reason:route.reason, latency:result.latency, totalTokens:result.totalTokens, inputTokens:result.inputTokens, outputTokens:result.outputTokens, costINR:result.costINR },...p].slice(0,300));
    } catch (err) {
      let em = err.message;
      if (em.includes("API_KEY")||em.includes("400")||em.includes("403")) em="API key invalid. Get a valid key from aistudio.google.com and update GEMINI_KEY in the file.";
      else if (em.includes("fetch")||em.includes("Failed")) em="Network error. Make sure the app is running via npm start and your internet is connected.";
      setMsgs(p=>[...p,{ id:`e${Date.now()}`, role:"assistant", content:`**Error:** ${em}`, ts:new Date().toISOString(), error:true }]);
    }
    setStreaming(false); setStreamTxt(""); setRouteInfo(null);
    setTimeout(() => inputRef.current?.focus(), 80);
  };

  const copy = async (id, text) => { await navigator.clipboard.writeText(text).catch(()=>{}); setCopied(id); setTimeout(()=>setCopied(null), 2000); };
  const clear = () => { if (confirm("Clear conversation? Memories are preserved.")) setMsgs([]); };

  const QA = [
    { l:"📊 Suggest strategy",      m:"Based on this customer profile and all previous interactions, suggest the best 3-step negotiation strategy to close this deal this week." },
    { l:"💰 Recommend discount",    m:"What discount and payment structure should I offer to close this deal while protecting our margins? Be specific with percentages." },
    { l:"🛡 Handle objections",     m:"Give me exact talking points and rebuttals to handle all their known objections. Make it ready to use in a call." },
    { l:"📄 Generate full proposal",m:"Generate a detailed business proposal with line-item pricing in ₹, payment milestones, SLA terms, and delivery timeline." },
    { l:"✉ Write follow-up email",  m:"Write a complete professional follow-up email I can send today. Make it persuasive, reference our last discussion, and include a clear CTA." },
    { l:"📋 Summarize conversation",m:"Summarize all key points from our conversation so far: what was discussed, decisions made, objections raised, and clear next steps." },
    { l:"🏆 Beat competitors",      m:"How do we differentiate against the competitors they mentioned? Give me a comparative analysis and 3 reasons why we win." },
    { l:"📅 Create closing plan",   m:"Create a detailed 2-week action plan with specific tasks and dates to close this deal by end of month." },
  ];

  if (!customer) {
    return (
      <div className="au" style={{ maxWidth:1100 }}>
        <div style={{ marginBottom:26 }}>
          <h1 style={{ fontSize:22, fontWeight:900, fontFamily:"'Space Grotesk',sans-serif", marginBottom:4 }} className="shimmer-text">AI Negotiation</h1>
          <p style={{ color:T.sub, fontSize:13 }}>Select a customer to start an AI-powered negotiation session</p>
        </div>
        <div className="card" style={{ padding:44, textAlign:"center", marginBottom:20 }}>
          <div style={{ width:64, height:64, borderRadius:18, background:`linear-gradient(135deg,${T.accent},${T.purple})`, display:"inline-flex", alignItems:"center", justifyContent:"center", marginBottom:16, boxShadow:`0 8px 32px rgba(124,111,255,0.3)` }}>
            <Ico d={P.chat} s={32} c="#fff" />
          </div>
          <p style={{ fontSize:16, color:T.sub, fontWeight:600, marginBottom:8 }}>No customer selected</p>
          <p style={{ fontSize:13, color:T.muted, marginBottom:28 }}>Pick a customer below or go to Customer Management</p>
          <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:12, maxWidth:520, margin:"0 auto" }}>
            {customers.slice(0,6).map(c => (
              <button key={c.id} className="card ch" style={{ padding:16, border:`1px solid ${T.border}`, background:"transparent", cursor:"pointer", textAlign:"left", transition:"all .2s" }} onClick={() => setAC(c)}>
                <Av name={c.name} size={36} />
                <div style={{ marginTop:10, fontWeight:600, fontSize:12 }}>{c.name}</div>
                <div style={{ fontSize:11, color:T.muted }}>{c.company}</div>
              </button>
            ))}
          </div>
          {customers.length>6 && <button className="btn bs" style={{ marginTop:16 }} onClick={() => nav("customers")}>View all {customers.length} customers</button>}
        </div>
      </div>
    );
  }

  return (
    <div style={{ display:"flex", height:"calc(100vh - 36px)", gap:14 }}>
      {/* Main chat */}
      <div style={{ flex:1, display:"flex", flexDirection:"column", minWidth:0 }}>
        <div className="card" style={{ padding:"12px 18px", marginBottom:14, display:"flex", alignItems:"center", gap:12, borderColor:T.borderMed }}>
          <Av name={customer.name} size={40} />
          <div style={{ flex:1, minWidth:0 }}>
            <div style={{ fontWeight:700, fontSize:14 }}>{customer.name}</div>
            <div style={{ fontSize:12, color:T.muted, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{customer.company} · {customer.budget||"Budget TBD"} · {mems.length} memories</div>
          </div>
          <div style={{ display:"flex", gap:7, flexShrink:0 }}>
            <Tag color={T.purple}>{mems.length} Hindsight</Tag>
            <StTag s={customer.status} />
            <button className="btn bg" style={{ padding:"5px 10px", fontSize:11 }} onClick={() => setAC(null)}>Change ×</button>
            <button className="btn bg" style={{ padding:"5px 10px", fontSize:11 }} onClick={clear}>Clear</button>
          </div>
        </div>

        <div style={{ flex:1, overflowY:"auto", display:"flex", flexDirection:"column", gap:14, paddingRight:2 }}>
          {msgs.length===0 && (
            <div>
              <div className="card" style={{ padding:"14px 18px", borderColor:T.accentBd, background:`linear-gradient(135deg, ${T.accentDim}, rgba(191,95,255,0.06))`, marginBottom:16 }}>
                <div style={{ display:"flex", gap:11, alignItems:"flex-start" }}>
                  <div style={{ width:28, height:28, borderRadius:8, background:`linear-gradient(135deg,${T.accent},${T.purple})`, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 }}>
                    <Ico d={P.brain} s={14} c="#fff" />
                  </div>
                  <div>
                    <div style={{ fontSize:13, fontWeight:600, color:T.accent, marginBottom:4 }}>
                      Hindsight {mems.length>0?`loaded ${mems.length} memories from past sessions`:"ready — no previous interactions found"}
                    </div>
                    <div style={{ fontSize:12, color:T.sub }}>
                      {mems.length>0?`Context available: ${mems.slice(0,3).map(m=>m.type).join(", ")}. AI will not repeat questions.`:"Profile data loaded. First interaction with this customer."}
                    </div>
                  </div>
                </div>
              </div>
              <div style={{ fontSize:11, color:T.muted, marginBottom:12, fontWeight:700, letterSpacing:".07em" }}>QUICK ACTIONS</div>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:9 }}>
                {QA.map(a => (
                  <button key={a.l} className="quick-btn" onClick={() => send(a.m)}>{a.l}</button>
                ))}
              </div>
            </div>
          )}

          {msgs.map(m => (
            <div key={m.id} className="ar" style={{ display:"flex", gap:10, alignItems:"flex-start", flexDirection:m.role==="user"?"row-reverse":"row" }}>
              {m.role==="assistant"
                ? <div style={{ width:32, height:32, borderRadius:9, background:`linear-gradient(135deg,${T.accent},${T.purple})`, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0, boxShadow:`0 3px 12px rgba(124,111,255,0.35)` }}><Ico d={P.zap} s={15} c="#fff" sw={2.5} /></div>
                : <Av name="You" size={32} />}
              <div style={{ maxWidth:m.role==="user"?"70%":"90%", minWidth:0 }}>
                <div style={{
                  background: m.role==="user" ? `linear-gradient(135deg,${T.accent},${T.purple})` : m.error ? T.redDim : T.card,
                  border: `1px solid ${m.role==="user"?"transparent":m.error?"rgba(255,91,107,0.2)":T.border}`,
                  borderRadius: m.role==="user" ? "14px 14px 4px 14px" : "4px 14px 14px 14px",
                  padding:"12px 16px",
                  boxShadow: m.role==="user" ? `0 4px 20px rgba(124,111,255,0.25)` : "none"
                }}>
                  {m.role==="user" ? <p style={{ fontSize:13, color:"#fff", lineHeight:1.75, margin:0 }}>{m.content}</p> : <Prose text={m.content} />}
                </div>
                <div style={{ display:"flex", alignItems:"center", gap:7, marginTop:5, flexWrap:"wrap" }}>
                  <span style={{ fontSize:11, color:T.muted }}>{new Date(m.ts).toLocaleTimeString("en-IN",{ hour:"2-digit", minute:"2-digit" })}</span>
                  {m.meta && (<><MdTag tier={m.meta.tier} /><Tag color={T.muted} bg="rgba(255,255,255,0.04)">{m.meta.latency}ms</Tag><Tag color={T.muted} bg="rgba(255,255,255,0.04)">{m.meta.tokens} tok</Tag><Tag color={T.amber} bg={T.amberDim}>₹{(m.meta.costINR||0).toFixed(4)}</Tag></>)}
                  {m.role==="assistant"&&!m.error && <button className="btn bg" style={{ padding:"2px 8px", fontSize:11 }} onClick={() => copy(m.id, m.content)}><Ico d={P.copy} s={11} />{copied===m.id?"Copied!":"Copy"}</button>}
                </div>
              </div>
            </div>
          ))}

          {streaming && (
            <div className="ar" style={{ display:"flex", gap:10, alignItems:"flex-start" }}>
              <div style={{ width:32, height:32, borderRadius:9, background:`linear-gradient(135deg,${T.accent},${T.purple})`, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0, boxShadow:`0 3px 12px rgba(124,111,255,0.35)` }}><Ico d={P.zap} s={15} c="#fff" sw={2.5} /></div>
              <div style={{ maxWidth:"90%", minWidth:0 }}>
                <div className="card" style={{ padding:"12px 16px" }}>
                  {streamTxt
                    ? <div className="tc"><Prose text={streamTxt} /></div>
                    : <div style={{ display:"flex", gap:9, alignItems:"center" }}><Spin s={14} /><span style={{ fontSize:12, color:T.sub }}>{routeInfo?`CascadeFlow → ${routeInfo.tier} (${routeInfo.reason})`:"Thinking…"}</span></div>}
                </div>
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input area */}
        <div className="card" style={{ padding:14, marginTop:12, borderColor:T.borderMed }}>
          <div style={{ display:"flex", gap:7, marginBottom:10, overflowX:"auto", paddingBottom:2 }}>
            {["Suggest discount","Handle objection","Write email","Generate proposal","Next steps","Compare competitors"].map(p => (
              <button key={p} className="chip" onClick={() => send(p)}>{p}</button>
            ))}
          </div>
          <div style={{ display:"flex", gap:10 }}>
            <textarea ref={inputRef} className="inp" style={{ flex:1, minHeight:60, maxHeight:180, resize:"none", lineHeight:1.6 }}
              placeholder={`Ask NegotiAI about ${customer.name}… (Enter to send, Shift+Enter for new line)`}
              value={input} onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key==="Enter"&&!e.shiftKey) { e.preventDefault(); send(); } }}
              disabled={streaming} />
            <button className="btn bp" style={{ padding:"0 20px", alignSelf:"flex-end", height:46, flexShrink:0 }} onClick={() => send()} disabled={streaming||!input.trim()}>
              {streaming ? <Spin s={16} /> : <Ico d={P.send} s={16} c="#fff" />}
            </button>
          </div>
        </div>
      </div>

      {/* Right sidebar */}
      <div style={{ width:256, flexShrink:0, display:"flex", flexDirection:"column", gap:12, overflowY:"auto" }}>
        <div className="card" style={{ padding:18 }}>
          <div style={{ fontSize:10, color:T.muted, fontWeight:700, letterSpacing:".1em", marginBottom:14 }}>CUSTOMER INTEL</div>
          {[[P.rupee,"Budget",customer.budget],[P.bldg,"Industry",customer.industry],[P.users,"Style",customer.style],[P.phone,"Phone",customer.phone],[P.mail,"Email",customer.email],[P.target,"Status",customer.status]].filter(([,,v])=>v).map(([ico,label,val]) => (
            <div key={label} style={{ display:"flex", gap:9, marginBottom:11 }}>
              <Ico d={ico} s={13} c={T.muted} />
              <div><div style={{ fontSize:10, color:T.muted, fontWeight:700 }}>{label.toUpperCase()}</div><div style={{ fontSize:12, color:T.sub, wordBreak:"break-word" }}>{val}</div></div>
            </div>
          ))}
        </div>
        {customer.objections && <div className="card" style={{ padding:16, borderColor:"rgba(255,91,107,0.2)", background:"rgba(255,91,107,0.03)" }}><div style={{ fontSize:10, color:T.red, fontWeight:700, letterSpacing:".1em", marginBottom:8 }}>⚠ OBJECTIONS</div><p style={{ fontSize:12, color:T.sub, lineHeight:1.75 }}>{customer.objections}</p></div>}
        {customer.competitors && <div className="card" style={{ padding:16 }}><div style={{ fontSize:10, color:T.amber, fontWeight:700, letterSpacing:".1em", marginBottom:8 }}>🎯 COMPETITORS</div><p style={{ fontSize:12, color:T.sub }}>{customer.competitors}</p></div>}
        {customer.requirements && <div className="card" style={{ padding:16 }}><div style={{ fontSize:10, color:T.blue, fontWeight:700, letterSpacing:".1em", marginBottom:8 }}>📋 REQUIREMENTS</div><p style={{ fontSize:12, color:T.sub, lineHeight:1.75 }}>{customer.requirements}</p></div>}
        <div className="card" style={{ padding:16 }}>
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:12 }}>
            <div style={{ fontSize:10, color:T.muted, fontWeight:700, letterSpacing:".1em" }}>HINDSIGHT MEMORY</div>
            <Tag color={T.purple}>{mems.length}</Tag>
          </div>
          {mems.length===0 && <p style={{ fontSize:12, color:T.muted }}>No memories yet. Significant exchanges are auto-saved.</p>}
          {mems.slice(0,5).map(m => {
            const cc = m.type==="meeting"?T.blue:m.type==="objection"?T.red:m.type==="proposal"?T.accent:m.type==="followup"?T.green:T.amber;
            return (
              <div key={m.id} style={{ display:"flex", gap:9, marginBottom:10, paddingBottom:10, borderBottom:`1px solid ${T.border}` }}>
                <div style={{ width:5, height:5, borderRadius:"50%", background:cc, marginTop:6, flexShrink:0, boxShadow:`0 0 6px ${cc}` }} />
                <div>
                  <Tag color={cc} style={{ marginBottom:4, fontSize:10 }}>{m.type}</Tag>
                  <div style={{ fontSize:11, color:T.muted, lineHeight:1.6, marginTop:3 }}>{m.content.slice(0,85)}…</div>
                  <div style={{ fontSize:10, color:T.muted, marginTop:3 }}>{new Date(m.date).toLocaleDateString("en-IN",{ day:"numeric", month:"short" })}</div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// MEMORY TIMELINE
// ═══════════════════════════════════════════════════════════════
function Memory({ customers }) {
  const [selId, setSelId] = useState(customers[0]?.id||null);
  const customer = customers.find(c=>c.id===selId);
  const mems = selId ? HS.load(selId) : [];
  const TM = { meeting:[T.blue,"👥 Meeting"], objection:[T.red,"⚠ Objection"], proposal:[T.accent,"📄 Proposal"], followup:[T.green,"✉ Follow-up"], note:[T.amber,"📝 Note"] };
  const allMems = customers.flatMap(c=>HS.load(c.id).map(m=>({ ...m, cn:c.name, cid:c.id }))).sort((a,b)=>new Date(b.date)-new Date(a.date));
  return (
    <div className="au" style={{ maxWidth:1100 }}>
      <div style={{ marginBottom:28 }}>
        <h1 style={{ fontSize:22, fontWeight:900, fontFamily:"'Space Grotesk',sans-serif", marginBottom:4 }} className="shimmer-text">Memory Timeline</h1>
        <p style={{ color:T.sub, fontSize:13 }}>Everything Hindsight has learned across all conversations</p>
      </div>
      <div style={{ display:"flex", gap:18 }}>
        <div style={{ width:240, flexShrink:0 }}>
          <div style={{ fontSize:10, color:T.muted, fontWeight:700, letterSpacing:".1em", marginBottom:12 }}>CUSTOMERS</div>
          {customers.map(c => {
            const cm = HS.load(c.id); const a = c.id===selId;
            return (
              <button key={c.id} onClick={() => setSelId(c.id)} style={{
                display:"flex", alignItems:"center", gap:10, width:"100%",
                padding:"11px 12px", borderRadius:10,
                border:`1px solid ${a?T.accentBd:T.border}`,
                background: a ? `linear-gradient(135deg,${T.accentDim},rgba(191,95,255,0.06))` : "transparent",
                cursor:"pointer", marginBottom:7, textAlign:"left",
                transition:"all .2s", fontFamily:"inherit",
              }}>
                <Av name={c.name} size={34} />
                <div style={{ flex:1, minWidth:0 }}>
                  <div style={{ fontSize:13, fontWeight:600, color:a?T.accent:T.text }}>{c.name}</div>
                  <div style={{ fontSize:11, color:T.muted }}>{cm.length} memories</div>
                </div>
                {cm.length>0 && <Tag color={T.purple}>{cm.length}</Tag>}
              </button>
            );
          })}
          {allMems.length>0 && (
            <>
              <div style={{ marginTop:22, fontSize:10, color:T.muted, fontWeight:700, letterSpacing:".1em", marginBottom:10 }}>ALL RECENT</div>
              {allMems.slice(0,7).map(m => {
                const [cc] = TM[m.type]||[T.amber];
                return (
                  <div key={m.id} onClick={() => setSelId(m.cid)} style={{ padding:"9px 11px", borderRadius:8, cursor:"pointer", marginBottom:5, border:`1px solid ${T.border}`, transition:"all .15s" }}
                    onMouseEnter={e => e.currentTarget.style.borderColor=T.borderMed}
                    onMouseLeave={e => e.currentTarget.style.borderColor=T.border}>
                    <div style={{ fontSize:11, color:T.sub, fontWeight:600 }}>{m.cn}</div>
                    <div style={{ fontSize:10, color:T.muted, marginTop:2 }}>{m.content.slice(0,55)}…</div>
                    <div style={{ width:5, height:5, borderRadius:"50%", background:cc, marginTop:5, boxShadow:`0 0 6px ${cc}` }} />
                  </div>
                );
              })}
            </>
          )}
        </div>
        <div style={{ flex:1, minWidth:0 }}>
          {!customer && <div className="card" style={{ padding:60, textAlign:"center", color:T.muted }}><Ico d={P.brain} s={36} c={T.muted} /><p style={{ marginTop:14 }}>Select a customer to view their memory timeline</p></div>}
          {customer && (
            <>
              <div className="card" style={{ padding:"18px 22px", marginBottom:18, display:"flex", alignItems:"center", gap:16 }}>
                <Av name={customer.name} size={50} />
                <div style={{ flex:1 }}>
                  <h2 style={{ fontSize:16, fontWeight:800 }}>{customer.name}</h2>
                  <div style={{ fontSize:13, color:T.muted }}>{customer.company} · {customer.city}</div>
                  <div style={{ display:"flex", gap:8, marginTop:9, flexWrap:"wrap" }}>
                    {[customer.budget, customer.industry, customer.style].filter(Boolean).map(v=><Tag key={v} color={T.sub}>{v}</Tag>)}
                  </div>
                </div>
                <div style={{ textAlign:"right" }}>
                  <div style={{ fontSize:30, fontWeight:900, color:T.accent, fontFamily:"'Space Grotesk',sans-serif" }}>{mems.length}</div>
                  <div style={{ fontSize:11, color:T.muted }}>total memories</div>
                </div>
              </div>
              {mems.length===0 && <div className="card" style={{ padding:60, textAlign:"center", color:T.muted }}><Ico d={P.brain} s={36} c={T.muted} /><p style={{ marginTop:14, color:T.sub }}>No memories yet</p><p style={{ fontSize:12, marginTop:6 }}>Start a negotiation session to build memory</p></div>}
              <div style={{ position:"relative" }}>
                {mems.length>0 && <div style={{ position:"absolute", left:20, top:0, bottom:0, width:2, background:`linear-gradient(to bottom,${T.accent},transparent)`, borderRadius:2 }}/>}
                {mems.map((m, idx) => {
                  const [cc, label] = TM[m.type]||[T.amber,"📝 Note"];
                  return (
                    <div key={m.id} className="au" style={{ display:"flex", gap:18, marginBottom:20, animationDelay:`${idx*.04}s` }}>
                      <div style={{ width:42, height:42, borderRadius:"50%", background:`${cc}18`, border:`2px solid ${cc}`, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0, zIndex:1, boxShadow:`0 0 12px ${cc}44` }}><div style={{ width:8, height:8, borderRadius:"50%", background:cc, boxShadow:`0 0 8px ${cc}` }}/></div>
                      <div className="card" style={{ flex:1, padding:"16px 20px" }}>
                        <div style={{ display:"flex", alignItems:"center", gap:9, marginBottom:9, flexWrap:"wrap" }}>
                          <Tag color={cc}>{label}</Tag>
                          <span style={{ fontSize:12, color:T.muted }}>{new Date(m.date).toLocaleDateString("en-IN",{ day:"numeric", month:"long", year:"numeric" })}</span>
                          {m.sentiment && <Tag color={m.sentiment==="positive"?T.green:T.sub}>{m.sentiment}</Tag>}
                        </div>
                        <p style={{ fontSize:13, color:T.sub, lineHeight:1.8 }}>{m.content}</p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// RUNTIME DASHBOARD
// ═══════════════════════════════════════════════════════════════
function Runtime({ auditLogs, setAuditLogs }) {
  const flash     = auditLogs.filter(l=>l.tier==="Flash");
  const pro       = auditLogs.filter(l=>l.tier==="Pro");
  const totalTok  = auditLogs.reduce((s,l)=>s+(l.totalTokens||0),0);
  const totalCost = auditLogs.reduce((s,l)=>s+(l.costINR||0),0);
  const avgLat    = auditLogs.length ? Math.round(auditLogs.reduce((s,l)=>s+(l.latency||0),0)/auditLogs.length) : 0;
  const savings   = flash.reduce((s,l)=>s+(l.totalTokens||0)*(0.00375-0.000075)/1000*83,0);
  const taskMap   = auditLogs.reduce((a,l)=>({ ...a, [l.taskType]:(a[l.taskType]||0)+1 }),{});
  const maxT      = Math.max(...Object.values(taskMap),1);
  const RULES = [
    { p:"proposal, contract, agreement", m:"Gemini 1.5 Pro",   r:"Complex structured generation"   },
    { p:"email, follow-up",              m:"Gemini 1.5 Pro",   r:"Nuanced tone & persuasion"       },
    { p:"objection handling",            m:"Gemini 1.5 Pro",   r:"Multi-step counter-argument"     },
    { p:"strategy, analysis",            m:"Gemini 1.5 Pro",   r:"Deep analytical reasoning"       },
    { p:"discount, negotiation",         m:"Gemini 1.5 Pro",   r:"Financial precision required"    },
    { p:"simple conversation (<250 ch)", m:"Gemini 1.5 Flash", r:"Cost-efficient, low latency"     },
  ];
  return (
    <div className="au" style={{ maxWidth:1100 }}>
      <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:28 }}>
        <div>
          <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:4 }}>
            <h1 style={{ fontSize:22, fontWeight:900, fontFamily:"'Space Grotesk',sans-serif" }} className="shimmer-text">Runtime Dashboard</h1>
            <Tag color={T.accent}>CascadeFlow</Tag>
          </div>
          <p style={{ color:T.sub, fontSize:13 }}>Real-time model routing, latency, token usage, cost, and full audit log</p>
        </div>
        {auditLogs.length>0 && <button className="btn bd" onClick={() => { if (confirm("Clear all audit logs?")) setAuditLogs([]); }}>Clear logs</button>}
      </div>

      <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:14, marginBottom:22 }}>
        {[["Total API Calls",auditLogs.length,T.accent],["Total Tokens",totalTok.toLocaleString("en-IN"),T.blue],["Total Cost",`₹${totalCost.toFixed(3)}`,T.amber],["Avg Latency",`${avgLat}ms`,T.green]].map(([l,v,c]) => (
          <div key={l} className="card card-3d" style={{ padding:"18px 20px", position:"relative", overflow:"hidden" }}>
            <div style={{ fontSize:24, fontWeight:900, color:c, fontFamily:"'Space Grotesk',sans-serif" }}>{v}</div>
            <div style={{ fontSize:11, color:T.muted, marginTop:4 }}>{l}</div>
            <div style={{ position:"absolute", top:0, right:0, width:60, height:60, borderRadius:"50%", background:c, filter:"blur(30px)", opacity:0.08, transform:"translate(30%,-30%)" }}/>
          </div>
        ))}
      </div>

      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:18, marginBottom:22 }}>
        <div className="card" style={{ padding:22 }}>
          <div style={{ fontWeight:700, fontSize:14, marginBottom:18 }}>⚡ CascadeFlow Model Split</div>
          {[[flash.length,"Gemini Flash — simple tasks",T.green],[pro.length,"Gemini Pro — complex tasks",T.accent]].map(([count,label,color]) => (
            <div key={label} style={{ marginBottom:16 }}>
              <div style={{ display:"flex", justifyContent:"space-between", marginBottom:7 }}>
                <span style={{ fontSize:12, color:T.sub }}>{label}</span>
                <span style={{ fontSize:13, fontWeight:700, color }}>{count} ({auditLogs.length?Math.round(count/auditLogs.length*100):0}%)</span>
              </div>
              <Bar val={count} max={Math.max(auditLogs.length,1)} color={color} />
            </div>
          ))}
          <div style={{ marginTop:18, padding:"14px 16px", background:T.greenDim, borderRadius:10, border:`1px solid rgba(0,229,160,0.18)` }}>
            <div style={{ fontSize:12, color:T.green, fontWeight:600 }}>💰 Savings vs all-Pro routing</div>
            <div style={{ fontSize:24, fontWeight:900, color:T.green, marginTop:5, fontFamily:"'Space Grotesk',sans-serif" }}>₹{savings.toFixed(3)}</div>
            <div style={{ fontSize:11, color:T.muted, marginTop:3 }}>{flash.length} calls routed to Flash instead of Pro</div>
          </div>
        </div>
        <div className="card" style={{ padding:22 }}>
          <div style={{ fontWeight:700, fontSize:14, marginBottom:18 }}>Task Type Distribution</div>
          {Object.keys(taskMap).length===0 && <p style={{ fontSize:13, color:T.muted }}>No tasks logged yet</p>}
          {Object.entries(taskMap).sort((a,b)=>b[1]-a[1]).map(([type,count]) => (
            <div key={type} style={{ display:"flex", alignItems:"center", gap:11, marginBottom:11 }}>
              <div style={{ fontSize:12, color:T.sub, textTransform:"capitalize", width:160, flexShrink:0 }}>{type.replace(/_/g," ")}</div>
              <Bar val={count} max={maxT} color={T.accent} />
              <span style={{ fontSize:12, fontWeight:700, width:20, textAlign:"right", color:T.accent }}>{count}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="card" style={{ padding:22, marginBottom:22 }}>
        <div style={{ fontWeight:700, fontSize:14, marginBottom:16 }}>CascadeFlow Routing Rules</div>
        <div style={{ display:"grid", gridTemplateColumns:"repeat(2,1fr)", gap:11 }}>
          {RULES.map(r => (
            <div key={r.p} style={{ display:"flex", gap:13, padding:"12px 15px", background:"rgba(124,111,255,0.03)", borderRadius:10, border:`1px solid ${T.border}`, transition:"border-color .2s" }}
              onMouseEnter={e => e.currentTarget.style.borderColor=T.borderMed}
              onMouseLeave={e => e.currentTarget.style.borderColor=T.border}>
              <div style={{ flex:1 }}><div style={{ fontSize:12, fontWeight:600 }}>{r.p}</div><div style={{ fontSize:11, color:T.muted, marginTop:3 }}>{r.r}</div></div>
              <Tag color={r.m.includes("Pro")?T.accent:T.green}>{r.m.includes("Pro")?"Pro":"Flash"}</Tag>
            </div>
          ))}
        </div>
      </div>

      <div className="card" style={{ padding:22 }}>
        <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:18 }}>
          <div style={{ fontWeight:700, fontSize:14 }}>Full Audit Log</div>
          <Tag color={T.muted}>{auditLogs.length} entries</Tag>
        </div>
        <div style={{ overflowX:"auto" }}>
          <table style={{ width:"100%", borderCollapse:"collapse", fontSize:12 }}>
            <thead>
              <tr style={{ borderBottom:`1px solid ${T.border}` }}>
                {["Time","Customer","Task","Model","Latency","In Tok","Out Tok","Cost ₹","Routing Reason"].map(h => (
                  <th key={h} style={{ textAlign:"left", padding:"9px 11px", color:T.muted, fontWeight:700, fontSize:11, letterSpacing:".05em", whiteSpace:"nowrap" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {auditLogs.slice(0,60).map(l => (
                <tr key={l.id} style={{ borderBottom:`1px solid ${T.border}`, transition:"background .15s" }}
                  onMouseEnter={e => e.currentTarget.style.background="rgba(124,111,255,0.04)"}
                  onMouseLeave={e => e.currentTarget.style.background=""}>
                  <td style={{ padding:"10px 11px", color:T.muted, whiteSpace:"nowrap" }}>{new Date(l.ts).toLocaleString("en-IN",{ day:"2-digit", month:"short", hour:"2-digit", minute:"2-digit" })}</td>
                  <td style={{ padding:"10px 11px", fontWeight:600 }}>{l.customer}</td>
                  <td style={{ padding:"10px 11px", color:T.sub, textTransform:"capitalize" }}>{(l.taskType||"").replace(/_/g," ")}</td>
                  <td style={{ padding:"10px 11px" }}><MdTag tier={l.tier} /></td>
                  <td style={{ padding:"10px 11px", color:l.latency>3000?T.amber:T.green, fontWeight:600 }}>{l.latency}ms</td>
                  <td style={{ padding:"10px 11px" }}>{(l.inputTokens||0).toLocaleString()}</td>
                  <td style={{ padding:"10px 11px" }}>{(l.outputTokens||0).toLocaleString()}</td>
                  <td style={{ padding:"10px 11px", color:T.amber }}>₹{(l.costINR||0).toFixed(4)}</td>
                  <td style={{ padding:"10px 11px", color:T.muted, maxWidth:200, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{l.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {auditLogs.length===0 && (
            <div style={{ textAlign:"center", padding:"60px 0", color:T.muted }}>
              <div style={{ width:56, height:56, borderRadius:16, background:T.accentDim, display:"inline-flex", alignItems:"center", justifyContent:"center", marginBottom:14 }}>
                <Ico d={P.zap} s={28} c={T.accent} />
              </div>
              <p style={{ marginTop:4 }}>No AI requests yet. Start a negotiation to see routing decisions.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// ROOT APP
// ═══════════════════════════════════════════════════════════════
export default function App() {
  const [splash,     setSplash]     = useState(true);
  const [user,       setUser]       = useState(null);
  const [page,       setPage]       = useState("dashboard");
  const [customers,  setCustomers]  = useDB("customers", SEED);
  const [auditLogs,  setAuditLogs]  = useDB("audit_logs", []);
  const [ac,         setAC]         = useState(null);

  useEffect(() => {
    if (!dbGet("customers") || dbGet("customers").length===0) setCustomers(SEED);
  }, []);

  function login(u)  { setUser(u); }
  function logout()  { setUser(null); setPage("dashboard"); setAC(null); }

  if (splash) return <><GS /><SplashScreen onDone={() => setSplash(false)} /></>;
  if (!user)  return <><GS /><LoginPage onLogin={login} /></>;

  const props = { customers, setCustomers, auditLogs, setAuditLogs, ac, setAC, nav:setPage };
  return (
    <>
      <GS />
      <div style={{ display:"flex", minHeight:"100vh" }}>
        <Sidebar page={page} nav={setPage} user={user} onLogout={logout} />
        <main style={{
          marginLeft:220, flex:1, padding:"28px 30px",
          minHeight:"100vh", overflowX:"hidden",
          background:`radial-gradient(ellipse at 10% 0%, rgba(124,111,255,0.04) 0%, transparent 50%)`,
        }}>
          {page==="dashboard" && <Dashboard {...props} setAC={setAC} />}
          {page==="customers" && <Customers {...props} setAC={setAC} />}
          {page==="chat"      && <Chat      {...props} />}
          {page==="memory"    && <Memory    customers={customers} />}
          {page==="runtime"   && <Runtime   auditLogs={auditLogs} setAuditLogs={setAuditLogs} />}
        </main>
      </div>
    </>
  );
}
