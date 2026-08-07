"""Generate utilitarian black/white/transparency web UI (Codex-style)."""
import pathlib

CSS = r"""
:root {
  --bg: #000;
  --surface: rgba(255,255,255,0.04);
  --surface2: rgba(255,255,255,0.07);
  --border: rgba(255,255,255,0.1);
  --border2: rgba(255,255,255,0.15);
  --text: #e0e0e0;
  --text2: #888;
  --muted: #555;
  --white: #fff;
  --mono: 'SF Mono', 'Cascadia Code', 'Fira Code', Consolas, monospace;
  --sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; overflow: hidden; }
body { font-family: var(--sans); background: var(--bg); color: var(--text); display: flex; font-size: 13px; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 3px; }
::-webkit-scrollbar-track { background: transparent; }
button { font: inherit; cursor: pointer; background: none; border: none; color: inherit; }
input, textarea, select { font: inherit; color: inherit; background: none; border: none; outline: none; }
#sidebar { width: 240px; flex: 0 0 240px; background: rgba(255,255,255,0.02); border-right: 1px solid var(--border); display: flex; flex-direction: column; transition: margin-left .15s, opacity .15s; }
#sidebar.collapsed { margin-left: -240px; opacity: 0; pointer-events: none; }
.sb-head { padding: 12px 14px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 8px; }
.sb-head .brand { flex: 1; font-family: var(--mono); font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; color: var(--white); font-weight: 600; }
.sb-new { border: 1px solid var(--border); padding: 4px 10px; font-size: 11px; font-family: var(--mono); color: var(--text2); transition: border-color .1s; }
.sb-new:hover { border-color: var(--white); color: var(--white); }
#projBar { padding: 8px 14px; border-bottom: 1px solid var(--border); display: flex; gap: 6px; align-items: center; }
#projBar select { flex: 1; background: var(--surface); border: 1px solid var(--border); color: var(--text); padding: 4px 6px; font-size: 11px; font-family: var(--mono); border-radius: 2px; }
#projBar select:focus { border-color: var(--white); }
#projBar select option { background: #111; color: var(--text); }
#projBar button { border: 1px solid var(--border); width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; font-size: 14px; color: var(--text2); font-family: var(--mono); }
#projBar button:hover { border-color: var(--white); color: var(--white); }
#convList { flex: 1; overflow-y: auto; padding: 6px; }
.conv { padding: 7px 10px; cursor: pointer; font-size: 12px; color: var(--text2); display: flex; gap: 6px; align-items: center; border: 1px solid transparent; font-family: var(--mono); transition: background .08s; }
.conv:hover { background: var(--surface); }
.conv.active { background: var(--surface2); color: var(--white); border-color: var(--border2); }
.conv .title { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.conv .del { opacity: 0; font-size: 13px; color: var(--muted); padding: 0 4px; }
.conv:hover .del { opacity: 1; }
.conv .del:hover { color: var(--white); }
#app { flex: 1; display: flex; flex-direction: column; min-width: 0; }
#topbar { flex: 0 0 auto; height: 40px; display: flex; align-items: center; gap: 8px; padding: 0 12px; border-bottom: 1px solid var(--border); background: rgba(255,255,255,0.02); }
.tb-btn { width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; color: var(--muted); border: 1px solid transparent; font-size: 12px; font-family: var(--mono); }
.tb-btn:hover { color: var(--white); border-color: var(--border); }
.tb-btn.active { color: var(--white); border-color: var(--white); }
.tb-btn svg { width: 14px; height: 14px; }
#modelSel { background: var(--surface); border: 1px solid var(--border); color: var(--text); padding: 3px 8px; font-size: 11px; font-family: var(--mono); border-radius: 2px; max-width: 200px; }
#modelSel option { background: #111; }
.tb-sep { width: 1px; height: 16px; background: var(--border); margin: 0 2px; }
#statusDot { width: 6px; height: 6px; border-radius: 50%; background: var(--white); transition: background .2s; }
#statusDot.off { background: #444; }
#statusDot.loading { background: var(--white); animation: pulse 1s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
#statusText { font-size: 10px; color: var(--muted); font-family: var(--mono); white-space: nowrap; }
#messages { flex: 1; overflow-y: auto; padding: 20px 16px 120px; scroll-behavior: smooth; }
.msg-wrap { max-width: 760px; margin: 0 auto; display: flex; flex-direction: column; gap: 12px; }
.msg { display: flex; gap: 8px; align-items: flex-start; }
.msg.user { flex-direction: row-reverse; }
.avatar { width: 22px; height: 22px; border: 1px solid var(--border); display: flex; align-items: center; justify-content: center; font-size: 9px; font-family: var(--mono); color: var(--muted); flex: 0 0 auto; }
.msg.assistant .avatar { border-color: var(--border2); color: var(--text2); }
.msg.user .avatar { border-color: var(--muted); color: var(--muted); }
.msg-body { max-width: 80%; min-width: 0; }
.bubble { padding: 8px 12px; font-size: 13px; line-height: 1.6; word-wrap: break-word; overflow-wrap: break-word; border: 1px solid var(--border); background: var(--surface); min-height: 20px; }
.msg.user .bubble { background: rgba(255,255,255,0.06); border-color: var(--border2); }
.msg.assistant .bubble { background: transparent; }
.msg-actions { display: flex; gap: 4px; margin-top: 4px; opacity: 0; transition: opacity .1s; }
.msg:hover .msg-actions { opacity: 1; }
.msg.user .msg-actions { justify-content: flex-end; }
.act-btn { font-size: 10px; color: var(--muted); padding: 2px 6px; font-family: var(--mono); border: 1px solid transparent; }
.act-btn:hover { color: var(--white); border-color: var(--border); }
.msg-meta { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 5px; }
.msg-meta .tag { font-size: 10px; color: var(--muted); font-family: var(--mono); border: 1px solid var(--border); padding: 1px 6px; }
.trace-box { margin-top: 6px; }
.trace-box pre { background: rgba(255,255,255,0.03); border: 1px solid var(--border); padding: 8px; font-size: 10px; line-height: 1.5; overflow: auto; max-height: 200px; font-family: var(--mono); color: var(--text2); display: none; }
.trace-box button { font-size: 10px; font-family: var(--mono); color: var(--muted); text-decoration: underline; padding: 0; }
.trace-box button:hover { color: var(--white); }
.bubble p { margin: 0 0 6px; } .bubble p:last-child { margin-bottom: 0; }
.bubble h1, .bubble h2, .bubble h3 { margin: 8px 0 4px; line-height: 1.3; font-weight: 600; }
.bubble h1 { font-size: 15px; } .bubble h2 { font-size: 14px; } .bubble h3 { font-size: 13px; }
.bubble ul, .bubble ol { margin: 4px 0 6px 18px; }
.bubble li { margin: 1px 0; }
.bubble a { color: var(--white); text-decoration: underline; text-underline-offset: 2px; }
.bubble code { background: rgba(255,255,255,0.06); padding: 1px 4px; font-family: var(--mono); font-size: 12px; }
.bubble blockquote { border-left: 2px solid var(--border2); margin: 4px 0; padding: 2px 10px; color: var(--text2); }
.bubble pre { position: relative; background: rgba(255,255,255,0.03); border: 1px solid var(--border); padding: 8px 10px 6px; font-size: 11px; line-height: 1.5; overflow: auto; max-height: 300px; margin: 4px 0; font-family: var(--mono); }
.bubble pre code { background: none; padding: 0; font-size: 11px; }
.bubble pre .lang { position: absolute; top: 4px; right: 36px; font-size: 9px; color: var(--muted); text-transform: uppercase; pointer-events: none; }
.bubble pre .copy-btn { position: absolute; top: 3px; right: 4px; border: 1px solid var(--border); color: var(--muted); font-size: 9px; padding: 1px 6px; font-family: var(--mono); opacity: 0; transition: opacity .1s; }
.bubble pre:hover .copy-btn { opacity: 1; }
.bubble pre .copy-btn:hover { color: var(--white); border-color: var(--white); }
.bubble pre .copy-btn.copied { color: var(--white); border-color: var(--white); }
.bubble hr { border: none; border-top: 1px solid var(--border); margin: 6px 0; }
.typing { display: flex; gap: 5px; padding: 10px 12px; align-items: center; }
.typing span { width: 6px; height: 6px; border-radius: 50%; background: var(--white); animation: typBounce 1s infinite; }
.typing span:nth-child(2) { animation-delay: .15s; }
.typing span:nth-child(3) { animation-delay: .3s; }
@keyframes typBounce { 0%,80%,100%{opacity:.2;transform:translateY(0)} 40%{opacity:1;transform:translateY(-4px)} }
.tl-bubble { font-family: var(--mono); font-size: 11.5px; line-height: 1.8; color: var(--text2); padding: 2px 12px 4px; }
.tl-line { display: flex; gap: 7px; align-items: baseline; max-width: 100%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.tl-line .tl-ico { flex: 0 0 auto; }
.tl-line.done { color: var(--muted); }
.tl-line.error { color: #f68; }
.tl-line.finished { color: #7bd88f; }
.tl-line.live .tl-ico { animation: tlPulse 1s infinite; }
@keyframes tlPulse { 0%,100% { opacity: .3; } 50% { opacity: 1; } }
.status-banner { background: rgba(255,255,255,0.08); border: 1px solid var(--border2); padding: 8px 12px; font-size: 11px; font-family: var(--mono); color: var(--text2); margin: 8px auto; max-width: 760px; text-align: center; }
#inputBar { flex: 0 0 auto; border-top: 1px solid var(--border); background: rgba(255,255,255,0.02); padding: 10px 16px 14px; }
.input-wrap { max-width: 760px; margin: 0 auto; position: relative; }
#attachments { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 6px; }
.chip { display: flex; align-items: center; gap: 4px; border: 1px solid var(--border); padding: 2px 6px; font-size: 10px; color: var(--text2); font-family: var(--mono); }
.chip .st { font-size: 9px; }
.chip button { font-size: 11px; color: var(--muted); padding: 0 2px; }
.chip button:hover { color: var(--white); }
#prompt { width: 100%; background: var(--surface); border: 1px solid var(--border); color: var(--white); padding: 10px 40px 10px 10px; font-size: 13px; line-height: 1.5; resize: none; max-height: 200px; min-height: 40px; font-family: var(--sans); border-radius: 2px; transition: border-color .15s; }
#prompt:focus { border-color: var(--white); }
#prompt::placeholder { color: var(--muted); }
#sendBtn { position: absolute; right: 6px; bottom: 6px; border: 1px solid var(--border); color: var(--muted); width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; font-family: var(--mono); transition: color .15s, border-color .15s, background .15s; }
#sendBtn:hover { color: var(--white); border-color: var(--white); }
#sendBtn:disabled { opacity: .25; cursor: not-allowed; }
#sendBtn.active-loading { border-color: var(--white); color: var(--white); animation: pulse 1s infinite; }
#sendBtn svg { width: 12px; height: 12px; }
#attachBtn { position: absolute; right: 38px; bottom: 6px; border: 1px solid var(--border); color: var(--muted); width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; }
#attachBtn:hover { color: var(--white); border-color: var(--white); }
#attachBtn svg { width: 12px; height: 12px; }
.empty { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; padding: 60px 20px; color: var(--muted); text-align: center; }
.empty .orb { width: 40px; height: 40px; border: 1px solid var(--border); display: flex; align-items: center; justify-content: center; font-family: var(--mono); font-size: 14px; color: var(--text2); }
.empty h2 { font-size: 14px; font-weight: 400; color: var(--text2); font-family: var(--mono); text-transform: uppercase; letter-spacing: 2px; }
.empty p { font-size: 12px; max-width: 400px; line-height: 1.5; color: var(--muted); }
.sugg-wrap { display: flex; gap: 6px; margin-top: 6px; flex-wrap: wrap; justify-content: center; }
.sugg { border: 1px solid var(--border); color: var(--text2); padding: 5px 10px; font-size: 11px; font-family: var(--mono); transition: border-color .1s; }
.sugg:hover { border-color: var(--white); color: var(--white); }
#fpanel { width: 260px; flex: 0 0 260px; background: rgba(255,255,255,0.02); border-left: 1px solid var(--border); display: flex; flex-direction: column; transition: margin-right .15s, opacity .15s; }
#fpanel.collapsed { margin-right: -260px; opacity: 0; pointer-events: none; }
.fp-head { padding: 10px 12px; border-bottom: 1px solid var(--border); display: flex; align-items: center; }
.fp-head span { font-family: var(--mono); font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: var(--text2); flex: 1; }
.fp-head button { color: var(--muted); font-size: 16px; }
.fp-head button:hover { color: var(--white); }
#fpList { flex: 1; overflow-y: auto; padding: 6px; }
.fp-item { display: flex; align-items: center; gap: 6px; padding: 6px 8px; font-size: 11px; color: var(--text2); font-family: var(--mono); border: 1px solid transparent; }
.fp-item:hover { background: var(--surface); border-color: var(--border); }
.fp-item input[type=checkbox] { accent-color: var(--white); flex: 0 0 auto; }
.fp-item .dot { width: 5px; height: 5px; border-radius: 50%; flex: 0 0 auto; }
.fp-item .dot.ok { background: var(--white); }
.fp-item .dot.err { background: #666; }
.fp-item .dot.idx { background: var(--muted); }
.fp-item .name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.fp-item .info { color: var(--muted); font-size: 9px; flex: 0 0 auto; }
#fpDrop { margin: 6px; padding: 20px; border: 1px dashed var(--border); text-align: center; color: var(--muted); font-size: 10px; font-family: var(--mono); transition: border-color .1s; }
#fpDrop.over { border-color: var(--white); color: var(--text2); }
#fpUpload { display: none; }
.overlay { position: fixed; inset: 0; background: rgba(0,0,0,.75); z-index: 100; display: none; align-items: center; justify-content: center; }
.overlay.open { display: flex; }
.modal { background: #0a0a0a; border: 1px solid var(--border2); padding: 20px; max-width: 380px; width: 90%; max-height: 80vh; overflow-y: auto; }
.modal h3 { font-size: 11px; font-family: var(--mono); text-transform: uppercase; letter-spacing: 1.5px; color: var(--text2); margin-bottom: 14px; }
.modal label { display: block; font-size: 10px; color: var(--muted); margin-bottom: 3px; margin-top: 10px; font-family: var(--mono); text-transform: uppercase; letter-spacing: 0.5px; }
.modal input[type=number], .modal input[type=text], .modal textarea { width: 100%; background: var(--surface); border: 1px solid var(--border); color: var(--text); padding: 6px 8px; font-size: 12px; font-family: var(--mono); border-radius: 2px; }
.modal input:focus, .modal textarea:focus { border-color: var(--white); }
.modal textarea { min-height: 50px; resize: vertical; }
.modal .actions { display: flex; gap: 6px; margin-top: 14px; justify-content: flex-end; }
.modal .btn { padding: 5px 12px; font-size: 11px; border: 1px solid var(--border); font-family: var(--mono); color: var(--text2); }
.modal .btn:hover { border-color: var(--white); color: var(--white); }
.modal .btn.primary { border-color: var(--white); color: var(--white); }
#searchBar { position: fixed; top: 40px; right: 0; left: 0; height: 36px; background: rgba(10,10,10,.95); border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 6px; padding: 0 12px; z-index: 90; transform: translateY(-100%); transition: transform .15s; backdrop-filter: blur(8px); }
#searchBar.open { transform: translateY(0); }
#searchInput { flex: 1; background: var(--surface); border: 1px solid var(--border); color: var(--text); padding: 4px 8px; font-size: 11px; max-width: 280px; font-family: var(--mono); border-radius: 2px; }
#searchInput:focus { border-color: var(--white); }
#searchCount { font-size: 10px; color: var(--muted); font-family: var(--mono); }
.sb-sbtn { border: 1px solid var(--border); color: var(--muted); width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; }
.sb-sbtn:hover { color: var(--white); border-color: var(--white); }
.sb-sbtn svg { width: 12px; height: 12px; }
#projDel { border: 1px solid var(--border); color: var(--muted); width: 22px; height: 22px; line-height: 1; flex: 0 0 22px; }
#projDel:hover { color: #ff6b6b; border-color: #ff6b6b; }
#projBanner { margin: 6px 10px; padding: 8px 10px; border: 1px solid #b8860b; background: rgba(255, 170, 0, 0.08); color: #ffd479; font-size: 11px; line-height: 1.5; }
#projBanner .reconnect { margin-top: 6px; border: 1px solid var(--border2); color: var(--white); padding: 4px 10px; font-size: 11px; }
#projBanner .reconnect:hover { border-color: var(--white); }
#projBanner.hidden { display: none; }
.sb-new-chat { margin: 8px 10px; padding: 6px 10px; border: 1px solid var(--border); color: var(--text2); font-family: var(--mono); font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
.sb-new-chat:hover { border-color: var(--white); color: var(--white); }
#projOpen svg { width: 13px; height: 13px; }
.status-banner.info { color: #7bd88f; border-color: #7bd88f; }
"""

HTML_BODY = r"""
<div id="sidebar">
  <div class="sb-head">
    <div class="brand">synapse</div>
    <button id="newChat" class="sb-new">+ new</button>
  </div>
  <div id="projBar">
    <select id="projSel" title=""></select>
    <button id="projOpen" title="Open existing project folder">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>
    </button>
    <button id="projNew" title="New project">+</button>
    <button id="projDel" title="Delete project">&times;</button>
  </div>
  <div id="projBanner" class="proj-banner hidden"></div>
  <button id="newChatBtn" class="sb-new-chat">new chat</button>
  <div id="convList"></div>
</div>
<div id="app">
  <div id="topbar">
    <button id="toggleSidebar" class="tb-btn" title="Sidebar">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 12h18M3 6h18M3 18h18"/></svg>
    </button>
    <select id="modelSel"><option value="">auto</option></select>
    <div class="tb-sep"></div>
    <button id="searchBtn" class="tb-btn" title="Search">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4-4"/></svg>
    </button>
    <button id="shortcutsBtn" class="tb-btn" title="Shortcuts">?</button>
    <button id="settingsBtn" class="tb-btn" title="Settings">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="2.5"/><path d="M12 1v2m0 18v2M4.22 4.22l1.42 1.42m12.72 12.72l1.42 1.42M1 12h2m18 0h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>
    </button>
    <div class="tb-sep"></div>
    <div id="statusDot"></div>
    <span id="statusText">...</span>
    <button id="toggleFiles" class="tb-btn" title="Files">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
    </button>
  </div>
  <div id="messages"><div class="msg-wrap" id="msgWrap"></div></div>
  <div id="inputBar">
    <div class="input-wrap">
      <div id="attachments"></div>
      <textarea id="prompt" rows="1" placeholder="..."></textarea>
      <button id="sendBtn" disabled title="Send">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
      </button>
      <button id="attachBtn" title="Attach">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
      </button>
    </div>
  </div>
</div>
<div id="fpanel" class="collapsed">
  <div class="fp-head"><span>files</span><button id="fpClose">&times;</button></div>
  <div id="fpList"></div>
  <div id="fpDrop">drop<input id="fpUpload" type="file" multiple></div>
</div>
<input id="pickInput" type="file" multiple hidden>
<div id="settingsOverlay" class="overlay">
  <div class="modal">
    <h3>settings</h3>
    <label>temperature <span id="setTempVal">0.7</span></label>
    <input id="setTemp" type="range" min="0" max="2" step="0.1" value="0.7">
    <label>max tokens</label>
    <input id="setMaxTok" type="number" placeholder="default" min="0">
    <label>system prompt</label>
    <textarea id="setSysPrompt" placeholder="optional"></textarea>
    <div class="actions">
      <button id="settingsCancel" class="btn">cancel</button>
      <button id="settingsSave" class="btn primary">save</button>
    </div>
  </div>
</div>
<div id="shortcutsOverlay" class="overlay">
  <div class="modal">
    <h3>shortcuts</h3>
    <div style="font-size:11px;line-height:2.2;color:var(--text2);font-family:var(--mono)">
      <div>enter &mdash; send</div>
      <div>shift+enter &mdash; newline</div>
      <div>ctrl+n &mdash; new chat</div>
      <div>ctrl+b &mdash; sidebar</div>
      <div>ctrl+f &mdash; search</div>
      <div>ctrl+/ &mdash; shortcuts</div>
      <div>ctrl+shift+o &mdash; files</div>
      <div>esc &mdash; close</div>
    </div>
    <div class="actions"><button id="scClose" class="btn primary">close</button></div>
  </div>
</div>
<div id="searchBar">
  <input id="searchInput" placeholder="search...">
  <span id="searchCount"></span>
  <button id="searchPrev" class="sb-sbtn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="18 15 12 9 6 15"/></svg></button>
  <button id="searchNext" class="sb-sbtn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="6 9 12 15 18 9"/></svg></button>
  <button id="searchClose" class="sb-sbtn">&times;</button>
</div>
"""

JS = r"""
const $ = id => document.getElementById(id);
const log = (...a) => console.log('[synapse]', ...a);
const err = (...a) => console.error('[synapse]', ...a);
const LS_SETTINGS = 'synapse_settings';
let settings = loadSettings();
let activeProjectId = null;
let activeChatId = null;
let projects = [];
let chats = [];
let messages = [];
let busy = false;
let attachedFiles = new Map();
const TL_ICON = { understand:'🧠', analyze:'🧭', read_workspace:'📂', read:'📄', list:'📂', search:'🔎', model:'🤖', generate:'✍', write:'💾', edit:'🔄', create_folder:'📁', rename:'🔀', delete:'🗑', verified:'🛡️', index:'🗂️', finished:'✅', error:'❌' };
function loadSettings() {
  try { return JSON.parse(localStorage.getItem(LS_SETTINGS)) || { temperature: 0.7, maxTokens: '', sysPrompt: '' }; }
  catch { return { temperature: 0.7, maxTokens: '', sysPrompt: '' }; }
}
function saveSettings() { localStorage.setItem(LS_SETTINGS, JSON.stringify(settings)); }
async function api(path, opts) {
  log('api', path, opts?.method || 'GET');
  const r = await fetch(path, opts);
  if (!r.ok) { const t = await r.text().catch(()=>''); throw new Error('HTTP ' + r.status + ': ' + t); }
  return r.json();
}
async function loadProjects() {
  try { projects = await api('/projects'); renderProjectSelector(); log('projects loaded', projects.length); }
  catch(e) { err('loadProjects', e); }
}
function renderProjectSelector() {
  const sel = $('projSel');
  const cur = activeProjectId;
  sel.innerHTML = '';
  projects.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.id;
    opt.textContent = p.name + (p.archived ? ' [archived]' : '') + (p.exists === false ? ' [missing]' : '');
    opt.title = 'workspace: ' + (p.workspace_path || '');
    sel.appendChild(opt);
  });
  if (cur) sel.value = cur;
  renderProjectBanner();
}
function renderProjectBanner() {
  const banner = $('projBanner');
  const p = projects.find(x => x.id === activeProjectId);
  if (!p || p.exists !== false) { banner.classList.add('hidden'); return; }
  banner.classList.remove('hidden');
  banner.innerHTML = '<div>Workspace folder not found:</div><div style="word-break:break-all;color:var(--text2)">' +
    esc(p.workspace_path || '') + '</div>';
  const btn = document.createElement('button');
  btn.className = 'reconnect';
  btn.textContent = 'locate folder';
  btn.addEventListener('click', async () => { await reconnectProject();
    await loadProjects(); });
  banner.appendChild(btn);
}
function baseName(p) {
  p = String(p || '').replace(/[\\/]+$/, '');
  const i = Math.max(p.lastIndexOf('\\'), p.lastIndexOf('/'));
  return i >= 0 ? p.slice(i + 1) : p;
}
function parentPath(p) {
  p = String(p || '').replace(/[\\/]+$/, '');
  const i = Math.max(p.lastIndexOf('\\'), p.lastIndexOf('/'));
  return i >= 0 ? p.slice(0, i + 1) : p;
}
async function pickProjectFolder() {
  try {
    const r = await fetch('/projects/pick', { method: 'POST' });
    if (r.ok) { const j = await r.json(); if (j.path) return j.path; }
  } catch (e) { err('pickProjectFolder', e); }
  const manual = prompt('native folder picker unavailable.\nenter the project folder path (created if it does not exist):');
  return (manual && manual.trim()) ? manual.trim() : null;
}
$('projSel').addEventListener('change', async () => { await switchProject($('projSel').value); });
$('projNew').addEventListener('click', async () => {
  // Creating a project asks ONE question: where should the folder live.
  // The project is named after the chosen folder; everything else is allowed.
  const folder = await pickProjectFolder();
  if (!folder) { showError('cancelled — no folder chosen'); return; }
  const body = { name: baseName(folder), parent_dir: parentPath(folder) };
  try {
    const info = await api('/projects', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    await loadProjects();
    await switchProject(info.id);
  } catch(e) { err('createProject', e); showError('create project failed: ' + e.message); }
});
function normPath(p) { return String(p || '').replace(/[\\/]+$/, '').toLowerCase(); }
async function openProject() {
  const folder = await pickProjectFolder();
  if (!folder) { showError('cancelled — no folder chosen'); return; }
  const existing = projects.find(p => p.exists !== false && normPath(p.workspace_path) === normPath(folder));
  if (existing) { notify('project "' + existing.name + '" is already open'); await switchProject(existing.id); return; }
  try {
    const info = await api('/projects', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ workspace_path: folder }) });
    await loadProjects();
    await switchProject(info.id);
    notify('opened "' + info.name + '" — indexing existing files…');
    await scanProject();
  } catch(e) { err('openProject', e); showError('open project failed: ' + e.message); }
}
$('projOpen').addEventListener('click', openProject);
async function scanProject() {
  if (!activeProjectId) return;
  try {
    const s = await api('/projects/' + activeProjectId + '/scan', { method: 'POST' });
    notify('scan: ' + s.files + ' file(s), ' + s.imported + ' imported, ' + s.index_jobs + ' index job(s)' + (s.language ? ', ' + s.language + (s.framework ? ' / ' + s.framework : '') : ''));
  } catch(e) { err('scanProject', e); showError('scan failed: ' + e.message); }
  refreshFiles();
}
function notify(msg) {
  const el = document.createElement('div');
  el.className = 'status-banner info';
  el.textContent = msg;
  $('msgWrap').appendChild(el); scrollToBottom();
  setTimeout(() => el.remove(), 6000);
}
$('projDel').addEventListener('click', async () => {
  const p = projects.find(x => x.id === activeProjectId);
  if (!p) return;
  const erase = confirm('Delete project "' + p.name + '" from Synapse?\n\nOK = remove from Synapse, KEEP workspace files.\nCancel = abort.');
  if (erase === false) return;
  const deleteFiles = confirm('ALSO permanently delete the workspace folder?\n\nOK = erase files on disk.\nCancel = keep the folder, just unregister.');
  try {
    const qs = deleteFiles ? '?delete_workspace=true' : '';
    await api('/projects/' + p.id + qs, { method: 'DELETE' });
    await loadProjects();
    if (activeProjectId === p.id) {
      activeProjectId = null;
      if (projects.length) await switchProject(projects[0].id);
      else { activeChatId = null; messages = []; renderMessages(); refreshFiles(); }
    }
  } catch(e) { err('deleteProject', e); showError('delete project failed: ' + e.message); }
});
async function reconnectProject() {
  const p = projects.find(x => x.id === activeProjectId);
  if (!p) return;
  const full = await pickParentFolder();
  if (!full) return;
  try {
    await api('/projects/' + p.id + '/reconnect', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ workspace_path: full }) });
    await loadProjects();
    renderProjectBanner();
  } catch(e) { err('reconnect', e); showError('reconnect failed: ' + e.message); }
}
async function switchProject(pid) {
  log('switchProject', pid);
  activeProjectId = pid;
  activeChatId = null;
  await api('/session', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ project_id: pid }) });
  await loadChats();
  if (chats.length) await switchChat(chats[0].id);
  else { activeChatId = null; messages = []; renderMessages(); }
  refreshFiles();
}
async function loadChats() {
  if (!activeProjectId) { chats = []; renderSidebar(); return; }
  try { chats = await api('/projects/' + activeProjectId + '/chats'); log('chats loaded', chats.length); }
  catch(e) { err('loadChats', e); chats = []; }
  renderSidebar();
}
async function switchChat(cid) {
  log('switchChat', cid);
  activeChatId = cid;
  try { messages = await api('/projects/' + activeProjectId + '/chats/' + cid + '/messages'); log('messages loaded', messages.length); }
  catch(e) { err('switchChat', e); messages = []; }
  await api('/session', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ project_id: activeProjectId, chat_id: cid }) });
  renderSidebar();
  renderMessages();
}
async function createChat() {
  if (!activeProjectId) { err('createChat: no activeProjectId'); return null; }
  try {
    log('createChat for', activeProjectId);
    const info = await api('/projects/' + activeProjectId + '/chats', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title: 'new' }) });
    log('chat created', info.id);
    await loadChats();
    await switchChat(info.id);
    return info.id;
  } catch(e) { err('createChat failed', e); return null; }
}
async function deleteChat(cid) {
  if (!activeProjectId) return;
  await api('/projects/' + activeProjectId + '/chats/' + cid, { method: 'DELETE' });
  if (activeChatId === cid) {
    activeChatId = null;
    await loadChats();
    if (chats.length) await switchChat(chats[0].id);
    else { messages = []; renderMessages(); }
  } else await loadChats();
}
function renderSidebar() {
  const list = $('convList');
  list.innerHTML = '';
  chats.forEach(c => {
    const el = document.createElement('div');
    el.className = 'conv' + (c.id === activeChatId ? ' active' : '');
    el.innerHTML = '<span class="title">' + esc(c.title) + '</span><button class="del">&times;</button>';
    el.querySelector('.title').addEventListener('click', () => switchChat(c.id));
    el.querySelector('.del').addEventListener('click', e => { e.stopPropagation(); deleteChat(c.id); });
    list.appendChild(el);
  });
}
$('newChat').addEventListener('click', createChat);
$('newChatBtn').addEventListener('click', createChat);
$('toggleSidebar').addEventListener('click', () => $('sidebar').classList.toggle('collapsed'));
function esc(s) { const d = document.createElement('div'); d.textContent = String(s ?? ''); return d.innerHTML; }
function mdInline(s) {
  return s.replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>').replace(/__([^_]+)__/g, '<strong>$1</strong>')
    .replace(/(^|[\s(])\*([^*\n]+)\*/g, '$1<em>$2</em>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
}
function renderMd(src) {
  const blocks = []; let s = esc(src);
  s = s.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => { const idx = blocks.length; blocks.push({ lang, code }); return '\x00B' + idx + '\x00'; });
  const out = []; const lines = s.split('\n'); let list = null, buf = [], inBlock = null;
  const flush = () => {
    if (list === 'ul') out.push('<ul>' + buf.map(x => '<li>' + mdInline(x) + '</li>').join('') + '</ul>');
    else if (list === 'ol') out.push('<ol>' + buf.map(x => '<li>' + mdInline(x) + '</li>').join('') + '</ol>');
    else if (inBlock === 'bq') out.push('<blockquote>' + buf.join('<br>') + '</blockquote>');
    else if (inBlock) out.push('<' + inBlock + '>' + buf.join('<br>') + '</' + inBlock + '>');
    else if (buf.length) out.push('<p>' + buf.join('<br>') + '</p>');
    buf = []; list = null; inBlock = null;
  };
  for (const raw of lines) {
    const hm = raw.match(/^(#{1,3})\s+(.*)$/); const ulm = raw.match(/^[-*]\s+(.*)$/);
    const olm = raw.match(/^\d+[.)]\s+(.*)$/); const qm = raw.match(/^>\s?(.*)$/);
    if (hm) { flush(); out.push('<h' + hm[1].length + '>' + mdInline(hm[2]) + '</h' + hm[1].length + '>'); }
    else if (ulm) { if (inBlock) flush(); if (list !== 'ul') { if (list) flush(); list = 'ul'; } buf.push(ulm[1]); }
    else if (olm) { if (inBlock) flush(); if (list !== 'ol') { if (list) flush(); list = 'ol'; } buf.push(olm[1]); }
    else if (qm) { if (list) flush(); if (inBlock !== 'bq') { if (inBlock) flush(); inBlock = 'bq'; } buf.push(qm[1]); }
    else if (/^\s*$/.test(raw)) flush();
    else if (/^---+$/.test(raw)) { flush(); out.push('<hr>'); }
    else { if (list) flush(); if (!inBlock) inBlock = 'p'; buf.push(raw); }
  }
  flush();
  return out.join('').replace(/\x00B(\d+)\x00/g, (_, i) => {
    const b = blocks[+i]; const langLabel = b.lang ? '<span class="lang">' + esc(b.lang) + '</span>' : '';
    return '<pre>' + langLabel + '<button class="copy-btn" onclick="copyCode(this)">copy</button><code>' + b.code + '</code></pre>';
  });
}
function copyCode(btn) {
  const code = btn.parentElement.querySelector('code');
  navigator.clipboard.writeText(code.textContent).then(() => {
    btn.textContent = 'copied'; btn.classList.add('copied');
    setTimeout(() => { btn.textContent = 'copy'; btn.classList.remove('copied'); }, 1200);
  });
}
function renderMessages() {
  const wrap = $('msgWrap'); wrap.innerHTML = '';
  if (!messages.length) {
    wrap.innerHTML = '<div class="empty" id="emptyState"><div class="orb">S</div><h2>ready</h2><p>type a message or drop files to begin.</p><div class="sugg-wrap"><button class="sugg">summarize workspace</button><button class="sugg">find todos</button><button class="sugg">explain architecture</button></div></div>';
    return;
  }
  messages.forEach((m, i) => appendMsgEl(m, i)); scrollToBottom();
}
function appendMsgEl(m, idx) {
  const wrap = $('msgWrap'); const emptyEl = $('emptyState'); if (emptyEl) emptyEl.remove();
  const el = document.createElement('div'); el.className = 'msg ' + m.role;
  const av = document.createElement('div'); av.className = 'avatar'; av.textContent = m.role === 'assistant' ? 'S' : 'U';
  const body = document.createElement('div'); body.className = 'msg-body';
  const bubble = document.createElement('div'); bubble.className = 'bubble';
  if (m.role === 'assistant') { bubble.innerHTML = m.content ? renderMd(m.content) : ''; }
  else bubble.textContent = m.content;
  body.appendChild(bubble);
  if (m.meta && m.meta.length) { const meta = document.createElement('div'); meta.className = 'msg-meta'; m.meta.forEach(t => { const s = document.createElement('span'); s.className = 'tag'; s.textContent = t; meta.appendChild(s); }); body.appendChild(meta); }
  if (m.trace) { const tb = document.createElement('div'); tb.className = 'trace-box'; tb.innerHTML = '<button>trace</button><pre>' + esc(JSON.stringify(m.trace, null, 2)) + '</pre>'; tb.querySelector('button').addEventListener('click', () => { const pre = tb.querySelector('pre'); const show = pre.style.display === 'none'; pre.style.display = show ? 'block' : 'none'; tb.querySelector('button').textContent = show ? 'hide' : 'trace'; }); body.appendChild(tb); }
  if (m.role === 'user') { const a = document.createElement('div'); a.className = 'msg-actions'; a.innerHTML = '<button class="act-btn">edit</button>'; a.querySelector('.act-btn').addEventListener('click', () => editMessage(idx)); body.appendChild(a); }
  if (m.role === 'assistant') { const a = document.createElement('div'); a.className = 'msg-actions'; a.innerHTML = '<button class="act-btn">copy</button><button class="act-btn">regen</button>'; a.querySelectorAll('.act-btn').forEach(b => { b.addEventListener('click', () => { if (b.textContent === 'copy') { navigator.clipboard.writeText(m.content); b.textContent = 'copied'; setTimeout(() => b.textContent = 'copy', 1200); } else regenerateFrom(idx); }); }); body.appendChild(a); }
  el.appendChild(av); el.appendChild(body); wrap.appendChild(el); return el;
}
function scrollToBottom() { const m = $('messages'); m.scrollTop = m.scrollHeight; }
function showTyping() {
  const wrap = $('msgWrap');
  const el = document.createElement('div');
  el.className = 'msg assistant';
  el.id = 'timelineBox';
  el.innerHTML = '<div class="avatar">S</div><div class="msg-body"><div class="bubble tl-bubble"></div></div>';
  wrap.appendChild(el);
  tlLastKey = '';
  scrollToBottom();
  log('timeline shown');
  return el;
}
function removeTyping() { const el = $('timelineBox'); if (el) el.remove(); }
let tlLastKey = '';
function tlAdd(kind, text) {
  const box = $('timelineBox'); if (!box) return;
  const key = kind + '|' + text;
  if (tlLastKey === key) return;
  tlLastKey = key;
  const bubble = box.querySelector('.tl-bubble');
  const line = document.createElement('div');
  line.className = 'tl-line live ' + (kind === 'error' ? 'error' : kind === 'finished' ? 'finished' : '');
  line.innerHTML = '<span class="tl-ico">' + (TL_ICON[kind] || '·') + '</span><span>' + esc(text || kind) + '</span>';
  bubble.appendChild(line);
  bubble.querySelectorAll('.tl-line.live').forEach(l => { if (l !== line) l.classList.remove('live'); });
  scrollToBottom();
}
function showError(msg) {
  err('UI error:', msg);
  const wrap = $('msgWrap');
  const el = document.createElement('div');
  el.className = 'status-banner';
  el.textContent = msg;
  wrap.appendChild(el);
  scrollToBottom();
  setTimeout(() => el.remove(), 8000);
}
function tlFinish() {
  const box = $('timelineBox'); if (!box) return;
  const bubble = box.querySelector('.tl-bubble');
  const live = bubble.querySelector('.tl-line.live');
  if (!live) return;
  live.classList.remove('live');
  live.classList.add('finished');
  live.querySelector('.tl-ico').textContent = TL_ICON.finished || '✅';
}
async function streamResponse(fullText, meta, trace) {
  removeTyping();
  return new Promise(resolve => {
    const wrap = $('msgWrap'); const idx = messages.length;
    const msg = { role: 'assistant', content: '', meta, trace }; messages.push(msg);
    const el = appendMsgEl(msg, idx); const bubble = el.querySelector('.bubble');
    let pos = 0;
    const cursor = document.createElement('span');
    cursor.style.cssText = 'display:inline-block;width:1px;height:1em;background:var(--white);margin-left:1px;animation:blink .8s infinite;vertical-align:text-bottom';
    bubble.appendChild(cursor);
    const speed = Math.max(3, Math.min(16, Math.ceil(fullText.length / 200)));
    function tick() {
      if (pos < fullText.length) { pos += speed; bubble.textContent = fullText.slice(0, pos); bubble.appendChild(cursor); scrollToBottom(); requestAnimationFrame(tick); }
      else { cursor.remove(); bubble.innerHTML = renderMd(fullText); resolve(); }
    }
    requestAnimationFrame(tick);
  });
}
const ta = $('prompt');
function autosize() { ta.style.height = 'auto'; ta.style.height = Math.min(ta.scrollHeight, 200) + 'px'; }
function updateSend() { $('sendBtn').disabled = busy || !ta.value.trim(); }
ta.addEventListener('input', () => { autosize(); updateSend(); });
ta.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); } });
$('sendBtn').addEventListener('click', () => { log('sendBtn clicked'); send(); });
document.addEventListener('click', e => { const b = e.target.closest('.sugg'); if (!b) return; ta.value = b.textContent.trim(); autosize(); updateSend(); send(); });

function setLoading(on) {
  busy = on;
  const btn = $('sendBtn');
  const dot = $('statusDot');
  if (on) {
    btn.disabled = true;
    btn.classList.add('active-loading');
    dot.className = 'loading';
    $('statusText').textContent = 'working...';
  } else {
    btn.classList.remove('active-loading');
    dot.className = '';
    updateSend();
    refreshStatus();
  }
}

async function send() {
  const prompt = ta.value.trim();
  log('send called, prompt length:', prompt.length, 'busy:', busy, 'activeProjectId:', activeProjectId, 'activeChatId:', activeChatId);
  if (!prompt || busy) { log('send aborted: empty or busy'); return; }
  if (!activeChatId) {
    log('no activeChatId, creating chat...');
    const cid = await createChat();
    if (!cid) { showError('failed to create chat'); return; }
    log('chat created:', cid);
  }
  const userMsg = { role: 'user', content: prompt, created_at: new Date().toISOString(), meta: [], trace: null, files: [] };
  messages.push(userMsg); appendMsgEl(userMsg, messages.length - 1);
  $('emptyState')?.remove(); ta.value = ''; autosize(); updateSend();
  setLoading(true);
  showTyping();
  try {
    const body = { prompt, files: [...attachedFiles.keys()], temperature: settings.temperature, project_id: activeProjectId, chat_id: activeChatId };
    if (settings.maxTokens) body.max_tokens = parseInt(settings.maxTokens) || undefined;
    const model = $('modelSel').value; if (model) body.model = model;
    log('sending request to /request/stream', body);
    const res = await fetch('/request/stream', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!res.ok) { const t = await res.text().catch(()=>''); throw new Error('HTTP ' + res.status + ': ' + t); }
    let final = null;
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    let done = false;
    while (!done) {
      const rd = await reader.read();
      if (rd.done) break;
      buf += dec.decode(rd.value, { stream: true });
      let sep;
      while ((sep = buf.indexOf('\n\n')) >= 0) {
        const chunk = buf.slice(0, sep); buf = buf.slice(sep + 2);
        const dl = chunk.split('\n').find(l => l.startsWith('data: '));
        if (!dl) continue;
        const ev = JSON.parse(dl.slice(6));
        if (ev.kind === 'done') { final = ev; done = true; break; }
        if (ev.kind === 'stream_error') throw new Error(ev.detail || 'request failed');
        tlAdd(ev.kind, ev.text);
      }
    }
    if (!final) throw new Error('stream closed without a response');
    tlFinish();
    const data = final.response;
    log('stream done, response keys:', Object.keys(data));
    setLoading(false);
    await streamResponse(data.response || '(no response)', final.meta || [], data.decision_trace);
  } catch (e) {
    removeTyping(); setLoading(false);
    showError('error: ' + e.message);
    const errMsg = { role: 'assistant', content: 'error: ' + e.message, created_at: new Date().toISOString(), meta: [], trace: null, files: [] };
    messages.push(errMsg); appendMsgEl(errMsg, messages.length - 1);
  }
  loadChats();
}
function regenerateFrom(msgIdx) { if (busy) return; messages.splice(msgIdx); renderMessages(); const lastUser = [...messages].reverse().find(m => m.role === 'user'); if (lastUser) { ta.value = lastUser.content; autosize(); updateSend(); } }
function editMessage(idx) { const m = messages[idx]; if (m.role !== 'user') return; const newText = prompt('edit:', m.content); if (newText !== null && newText.trim()) { messages.splice(idx); renderMessages(); ta.value = newText.trim(); autosize(); updateSend(); } }
function refreshModels() {
  fetch('/models').then(r => r.json()).then(models => {
    const sel = $('modelSel'); const cur = sel.value; sel.innerHTML = '<option value="">auto</option>';
    models.forEach(m => { const opt = document.createElement('option'); opt.value = m.id; opt.textContent = m.id + (m.available ? '' : ' *'); opt.disabled = !m.available; sel.appendChild(opt); }); sel.value = cur;
  }).catch(e => err('refreshModels', e));
}
$('toggleFiles').addEventListener('click', () => { const p = $('fpanel'); p.classList.toggle('collapsed'); $('toggleFiles').classList.toggle('active', !p.classList.contains('collapsed')); });
$('fpClose').addEventListener('click', () => { $('fpanel').classList.add('collapsed'); $('toggleFiles').classList.remove('active'); });
function refreshFiles() {
  const qs = activeProjectId ? '?project_id=' + encodeURIComponent(activeProjectId) : '';
  fetch('/workspace' + qs).then(r => r.json()).then(w => {
    const list = $('fpList');
    if (!w.files.length) { list.innerHTML = '<div style="padding:16px;text-align:center;color:var(--muted);font-size:10px;font-family:var(--mono)">empty</div>'; return; }
    list.innerHTML = '';
    w.files.forEach(f => { const el = document.createElement('div'); el.className = 'fp-item'; const checked = attachedFiles.has(f.id) ? 'checked' : ''; const dotClass = f.status === 'indexed' ? 'ok' : f.status === 'failed' ? 'err' : 'idx'; el.innerHTML = '<input type="checkbox" ' + checked + '><span class="dot ' + dotClass + '"></span><span class="name">' + esc(f.name) + '</span><span class="info">' + f.indexed_chunks + '</span>'; el.querySelector('input').addEventListener('change', e => { if (e.target.checked) attachedFiles.set(f.id, { name: f.name, status: f.status }); else attachedFiles.delete(f.id); renderChips(); }); list.appendChild(el); });
    renderChips();
  }).catch(e => err('refreshFiles', e));
}
function renderChips() {
  const box = $('attachments'); box.innerHTML = '';
  attachedFiles.forEach((info, id) => { const chip = document.createElement('div'); chip.className = 'chip'; chip.innerHTML = '<span>' + esc(info.name) + '</span><span class="st">' + info.status + '</span><button>&times;</button>'; chip.querySelector('button').addEventListener('click', () => { attachedFiles.delete(id); renderChips(); refreshFiles(); }); box.appendChild(chip); });
}
function uploadFile(file) {
  const fd = new FormData(); fd.append('file', file);
  const qs = activeProjectId ? '?project_id=' + encodeURIComponent(activeProjectId) : '';
  fetch('/files' + qs, { method: 'POST', body: fd }).then(r => r.json()).then(res => { attachedFiles.set(res.file.id, { name: res.file.name, status: res.file.status }); renderChips(); refreshFiles(); $('fpanel').classList.remove('collapsed'); $('toggleFiles').classList.add('active'); }).catch(e => err('uploadFile', e));
}
$('attachBtn').addEventListener('click', () => $('pickInput').click());
$('pickInput').addEventListener('change', e => { [...e.target.files].forEach(uploadFile); e.target.value = ''; });
$('fpDrop').addEventListener('dragover', e => { e.preventDefault(); $('fpDrop').classList.add('over'); });
$('fpDrop').addEventListener('dragleave', () => $('fpDrop').classList.remove('over'));
$('fpDrop').addEventListener('drop', e => { e.preventDefault(); $('fpDrop').classList.remove('over'); [...(e.dataTransfer?.files || [])].forEach(uploadFile); });
$('fpUpload').addEventListener('change', e => { [...e.target.files].forEach(uploadFile); e.target.value = ''; });
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', e => { e.preventDefault(); [...(e.dataTransfer?.files || [])].forEach(uploadFile); });
$('settingsBtn').addEventListener('click', () => { $('setTemp').value = settings.temperature; $('setTempVal').textContent = settings.temperature; $('setMaxTok').value = settings.maxTokens || ''; $('setSysPrompt').value = settings.sysPrompt || ''; $('settingsOverlay').classList.add('open'); });
$('setTemp').addEventListener('input', () => { $('setTempVal').textContent = $('setTemp').value; });
$('settingsCancel').addEventListener('click', () => $('settingsOverlay').classList.remove('open'));
$('settingsSave').addEventListener('click', () => { settings.temperature = parseFloat($('setTemp').value) || 0.7; settings.maxTokens = $('setMaxTok').value; settings.sysPrompt = $('setSysPrompt').value; saveSettings(); $('settingsOverlay').classList.remove('open'); });
let searchIdx = -1;
function openSearch() { $('searchBar').classList.add('open'); $('searchInput').focus(); }
function closeSearch() { $('searchBar').classList.remove('open'); clearHighlights(); searchIdx = -1; }
function clearHighlights() { document.querySelectorAll('mark.search-hl').forEach(m => { m.replaceWith(document.createTextNode(m.textContent)); }); }
function doSearch() { clearHighlights(); const q = $('searchInput').value.trim(); if (!q) { $('searchCount').textContent = ''; return; } const walker = document.createTreeWalker($('msgWrap'), NodeFilter.SHOW_TEXT); const matches = []; while (walker.nextNode()) { const node = walker.currentNode; const idx = node.textContent.toLowerCase().indexOf(q.toLowerCase()); if (idx >= 0) matches.push({ node, idx }); } $('searchCount').textContent = matches.length ? matches.length + ' found' : '0'; if (matches.length) { searchIdx = 0; highlightMatch(matches[searchIdx]); } }
function highlightMatch(m) { document.querySelectorAll('mark.current').forEach(el => el.classList.remove('current')); const range = document.createRange(); range.setStart(m.node, m.idx); range.setEnd(m.node, m.idx + $('searchInput').value.length); const mark = document.createElement('mark'); mark.className = 'search-hl current'; range.surroundContents(mark); mark.scrollIntoView({ behavior: 'smooth', block: 'center' }); }
$('searchBtn').addEventListener('click', openSearch); $('searchClose').addEventListener('click', closeSearch); $('searchInput').addEventListener('input', doSearch); $('searchNext').addEventListener('click', doSearch); $('searchPrev').addEventListener('click', doSearch);
$('shortcutsBtn').addEventListener('click', () => $('shortcutsOverlay').classList.add('open'));
$('scClose').addEventListener('click', () => $('shortcutsOverlay').classList.remove('open'));
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') { if ($('searchBar').classList.contains('open')) closeSearch(); else if ($('settingsOverlay').classList.contains('open')) $('settingsOverlay').classList.remove('open'); else if ($('shortcutsOverlay').classList.contains('open')) $('shortcutsOverlay').classList.remove('open'); }
  if (e.ctrlKey && e.key === 'n') { e.preventDefault(); createChat(); }
  if (e.ctrlKey && e.key === 'b') { e.preventDefault(); $('sidebar').classList.toggle('collapsed'); }
  if (e.ctrlKey && e.key === 'f') { e.preventDefault(); openSearch(); }
  if (e.ctrlKey && e.key === '/') { e.preventDefault(); $('shortcutsOverlay').classList.toggle('open'); }
  if (e.ctrlKey && e.shiftKey && e.key === 'O') { e.preventDefault(); $('fpanel').classList.toggle('collapsed'); $('toggleFiles').classList.toggle('active', !$('fpanel').classList.contains('collapsed')); }
});
function refreshStatus() {
  fetch('/status').then(r => r.json()).then(st => {
    const ready = st.providers.some(p => p.ready); $('statusDot').className = ready ? '' : 'off';
    const names = st.providers.map(p => p.provider_id).join(', ');
    $('statusText').textContent = (names || 'none') + ' / ' + st.registry_count;
  }).catch(e => { $('statusDot').className = 'off'; $('statusText').textContent = 'offline'; err('refreshStatus', e); });
}
async function boot() {
  log('boot starting');
  try {
    const session = await api('/session');
    log('session loaded', session);
    if (session.project_id) activeProjectId = session.project_id;
    if (session.chat_id) activeChatId = session.chat_id;
  } catch(e) { err('session load failed', e); }
  await loadProjects();
  if (!activeProjectId && projects.length) activeProjectId = projects[0].id;
  if (activeProjectId) { $('projSel').value = activeProjectId; await loadChats(); }
  if (activeChatId && chats.some(c => c.id === activeChatId)) await switchChat(activeChatId);
  else if (chats.length) await switchChat(chats[0].id);
  else { messages = []; renderMessages(); }
  refreshModels(); refreshStatus(); refreshFiles();
  log('boot complete, activeProjectId:', activeProjectId, 'activeChatId:', activeChatId);
  setInterval(refreshStatus, 15000); setInterval(refreshFiles, 5000);
}
boot().catch(e => { err('boot failed', e); showError('boot failed: ' + e.message); });
"""

PAGE = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>synapse</title>
<style>{CSS}</style>
</head>
<body>
{HTML_BODY}
<script>{JS}</script>
</body>
</html>"""


def render_page() -> str:
    return PAGE
