from datetime import datetime
from io import BytesIO
from pathlib import Path
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter

from src.preprocessing import preprocess
from src.spectral_simulation import pseudo_cube, false_color, WAVELENGTHS
from src.feature_extraction import extract_features, signature
from src.classifier import get_model
from src.risk_scoring import score_risk, risk_band
from src.conveyor_simulation import route_for

ROOT = Path(__file__).parent
st.set_page_config(page_title="SpectraSafe | Inspection Console", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
.stApp {background:#f4f7fb;color:#172b4d;font-family:'DM Sans',sans-serif}
[data-testid="stSidebar"] {background:#10243e}
[data-testid="stSidebar"] * {color:#edf4fb}
h1,h2,h3 {font-family:'Manrope',sans-serif;color:#142943}
.hero {background:linear-gradient(115deg,#10243e,#193e66);padding:27px 32px;border-radius:18px;color:white;margin-bottom:20px;box-shadow:0 12px 34px #12243a18}
.hero h1 {color:white;font-size:36px;margin:0}.hero p {color:#c5d5e5;margin:4px 0 0}.subline {color:#9fb8d3!important;font-size:13px;letter-spacing:.04em}
.badge {display:inline-block;background:#d9efff;color:#14558a;padding:7px 12px;border-radius:20px;font-size:11px;font-weight:800;letter-spacing:.08em}
.card {background:white;border:1px solid #e3eaf2;border-radius:15px;padding:18px 20px;box-shadow:0 5px 20px #21395608;margin-bottom:12px}
.kicker {color:#73859a;font-size:11px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;margin-bottom:10px}
.metric {font-size:34px;font-weight:800;color:#142943;line-height:1.1}.muted {color:#718198;font-size:13px}
.decision {border-radius:16px;padding:20px;text-align:center;color:white;font-weight:800;font-size:27px;letter-spacing:.05em}
.decision.pass {background:linear-gradient(120deg,#137f61,#21a67d)}.decision.flag {background:linear-gradient(120deg,#b83b42,#e36354)}
.stage {display:inline-block;background:#eef4fa;border:1px solid #d9e4ef;border-radius:10px;padding:10px 13px;margin:4px;color:#35516e;font-size:12px;font-weight:700}
.foot {color:#708096;font-size:11px;text-align:center;padding:20px}
.conveyor {background:#152d48;border-radius:15px;padding:16px;color:white;text-align:center}
.ss3d-shell {background:linear-gradient(140deg,#0d1e32,#193b5b);border-radius:18px;padding:18px 22px;color:#eef7ff;box-shadow:0 12px 30px #13273d20;overflow:hidden}
.ss3d-head {display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:8px}.ss3d-head strong {font-size:15px;letter-spacing:.09em}.ss3d-tag {font-size:10px;letter-spacing:.1em;color:#a9c4dc;border:1px solid #55718d;border-radius:20px;padding:6px 10px}
.ss3d-scene {position:relative;height:265px;overflow:hidden;border-radius:14px;background:radial-gradient(ellipse at 50% 80%,#24496b 0%,#142d48 56%,#10243a 100%);perspective:900px}
.ss3d-grid {position:absolute;inset:48% -10% -32%;transform:rotateX(59deg);background-image:linear-gradient(#82b5df17 1px,transparent 1px),linear-gradient(90deg,#82b5df17 1px,transparent 1px);background-size:36px 36px;mask-image:linear-gradient(transparent,#000 28%)}
.ss3d-belt {position:absolute;left:3%;right:3%;bottom:24px;height:103px;transform:rotateX(53deg);transform-origin:bottom center;border:2px solid #446987;border-radius:10px;background:repeating-linear-gradient(90deg,#273c50 0 40px,#34516a 40px 44px);box-shadow:0 17px 24px #06111f99;animation:belt-shift .42s linear infinite}
@keyframes belt-shift {to{background-position:44px 0}}
.ss3d-rail {position:absolute;left:3%;right:3%;bottom:76px;height:3px;background:#80acc9;box-shadow:0 15px #80acc966}
.ss3d-station {position:absolute;top:24px;transform:translateX(-50%);text-align:center;z-index:2;width:140px}.ss3d-station .ss3d-icon {margin:auto auto 7px;width:58px;height:54px;border:2px solid #7295b3;border-radius:8px;background:linear-gradient(145deg,#456988,#1e3853);box-shadow:8px 9px 0 #0a1726;display:flex;align-items:center;justify-content:center;font-size:22px;color:#dff4ff}.ss3d-station b {display:block;font-size:10px;letter-spacing:.07em;color:#e1eef8}.ss3d-station small {color:#9bb5ca;font-size:9px}.ss3d-station .beam {height:36px;width:3px;background:linear-gradient(#58d9ff99,transparent);position:absolute;left:50%;top:54px;box-shadow:0 0 13px #48cfff}
.ss3d-s1{left:12%}.ss3d-s2{left:34%}.ss3d-s3{left:56%}.ss3d-s4{left:78%}
.ss3d-grain {position:absolute;z-index:5;left:2%;bottom:74px;width:50px;height:27px;border-radius:60% 48% 55% 45%;background:radial-gradient(ellipse at 35% 28%,#ffe5a2 0 10%,#ce9b54 35%,#80582e 78%);border:2px solid #f6d89b;box-shadow:0 6px 8px #07111caa;animation:grain-pass 6.4s cubic-bezier(.45,0,.55,1) infinite}
.ss3d-grain:after {content:'DEMO';position:absolute;top:6px;left:10px;color:#563d21;font:bold 7px sans-serif;letter-spacing:.04em}
@keyframes grain-pass {0%{left:2%;bottom:73px;transform:rotate(0)}18%{left:17%;bottom:73px;transform:rotate(8deg)}38%{left:37%;bottom:73px;transform:rotate(-6deg)}57%{left:56%;bottom:73px;transform:rotate(5deg)}76%{left:74%;bottom:73px;transform:rotate(-3deg)}92%,100%{left:91%;bottom:73px;transform:rotate(7deg)}}
@keyframes grain-flag {0%{left:2%;bottom:73px;transform:rotate(0)}18%{left:17%;bottom:73px;transform:rotate(8deg)}38%{left:37%;bottom:73px;transform:rotate(-6deg)}57%{left:56%;bottom:73px;transform:rotate(5deg)}76%{left:74%;bottom:73px;transform:rotate(-3deg)}87%{left:77%;bottom:35px;transform:rotate(40deg)}100%{left:83%;bottom:24px;transform:rotate(90deg)}}
.ss3d-shell.pass .ss3d-grain {animation-name:grain-pass}.ss3d-shell.flag .ss3d-grain {animation-name:grain-flag}
.ss3d-route {position:absolute;right:3%;bottom:6px;display:flex;gap:8px;z-index:6}.ss3d-bin {background:#18334e;border:1px solid #54728e;border-radius:8px;padding:6px 10px;color:#ccdeec;font-size:9px;letter-spacing:.08em}.ss3d-bin.active {background:#126b55;border-color:#56d2a9;color:#e4fff6}.ss3d-shell.flag .ss3d-bin.active {background:#8e3436;border-color:#ff8581;color:#fff1ef}
.ss3d-steps {display:flex;justify-content:space-between;gap:7px;margin-top:12px;flex-wrap:wrap}.ss3d-step {flex:1;min-width:110px;border-radius:8px;background:#1a3854;border:1px solid #345774;padding:8px 10px;color:#d7e5f1;font-size:10px}.ss3d-step em {display:block;font-style:normal;color:#72caff;font-size:9px;letter-spacing:.08em;margin-bottom:3px}.ss3d-step:last-child{border-color:#6485a2}
.ss3d-step.active {background:#22577a;border-color:#5fd5ff;box-shadow:0 0 15px #3cbfff35}.ss3d-step.active em {color:#9de8ff}
.ss3d-shell.paused .ss3d-belt,.ss3d-shell.paused .direction,.ss3d-shell.paused .data-packets i,.ss3d-shell.paused .scan-line {animation-play-state:paused!important}
.line-status {background:#0c1a2b;color:#e5f1fa;border:1px solid #28435c;border-radius:14px;padding:15px 20px;margin:10px 0 14px;box-shadow:0 10px 25px #0d203018}.line-status h2 {margin:0;color:#f4f8fc;font-size:20px;letter-spacing:.08em}.line-status small {color:#9eb6cc;letter-spacing:.1em}.line-status-chips {display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}.line-status-chips span {background:#142b42;border:1px solid #35536e;color:#c5d8e7;border-radius:7px;padding:7px 9px;font-size:10px;letter-spacing:.04em}
.ss3d-scene.line-scene {height:clamp(340px,38vw,540px);background:radial-gradient(ellipse at 50% 82%,#203e57 0%,#11273b 55%,#0d1d30 100%);border:1px solid #425e73;box-shadow:inset 0 0 45px #06111b55,0 15px 32px #05101e66}
.line-scene .line-backdrop {position:absolute;inset:0;z-index:1;overflow:hidden;border-radius:inherit}
.line-scene .line-backdrop svg {display:block;width:100%;height:100%}
.line-scene .line-backdrop img {display:block;width:100%;height:100%;object-fit:cover;object-position:center 54%;filter:saturate(.91) contrast(1.04)}
.line-scene .line-backdrop:after {content:"";position:absolute;inset:0;background:linear-gradient(180deg,#07182708 35%,#07182735 100%);pointer-events:none}
.line-scene .ss3d-grid,.line-scene .ss3d-rail,.line-scene .ss3d-belt,.line-scene .roller-row {display:none}
.line-scene .line-station {display:grid;place-items:center;width:32px;height:32px;padding:0;transform:translate(-50%,-50%);text-align:center;border:2px solid #e8fbff;border-radius:50%;background:#087e78;box-shadow:0 3px 12px #06111dbb,0 0 0 4px #50ddd33b;backdrop-filter:blur(4px);z-index:7}
.line-scene .line-station .ss3d-icon,.line-scene .line-station b,.line-scene .line-station small {display:none}.line-scene .line-pin-number {font:700 10px 'DM Sans',sans-serif;color:#f0f7fc;line-height:1}
.line-scene .line-station.hot {border-color:#ffffff;background:#10a99d;box-shadow:0 0 0 5px #62d5ff44,0 0 18px #32c8ff99}
.line-scene .line-station.light-on .beam {opacity:1}
.line-scene .beam {position:absolute;opacity:0;left:50%;top:100%;width:3px;height:130px;background:linear-gradient(#77e7ff99,#5acfff00);transition:opacity .25s;pointer-events:none}
.line-scene .line-station.light-on {border-color:#75e8ff;background:#126b88;box-shadow:0 0 0 7px #57dfff22,0 0 22px #36d5ff99}
.line-scene .moving-sample {position:absolute;z-index:8;width:57px;height:31px;top:var(--product-y);border-radius:60% 48% 55% 45%;background:radial-gradient(ellipse at 35% 28%,#ffe5a2 0 10%,#ce9b54 35%,#80582e 78%);border:2px solid #f6d89b;box-shadow:0 8px 13px #07111caa,0 0 18px #ffda8e88;transform:translate(-50%,-50%) rotate(-6deg);transition:left .14s linear,top .14s linear}
.line-scene .moving-sample:after {content:'SAMPLE';position:absolute;top:8px;left:10px;color:#563d21;font:bold 7px sans-serif;letter-spacing:.04em}
.line-scene .scan-line {position:absolute;z-index:9;left:var(--product-x);top:var(--product-y);width:3px;height:42px;background:#71eaff;opacity:0;box-shadow:0 0 16px #3edbff;transition:left .75s,top .75s;animation:scan-sweep .85s ease-in-out infinite alternate}
.line-scene.scan-active .scan-line {opacity:1}
@keyframes scan-sweep {from{transform:translateX(-19px)}to{transform:translateX(19px)}}
.line-scene .sim-sort-gate {position:absolute;z-index:10;left:86%;top:39%;width:58px;height:54px;pointer-events:none;transform:translate(-50%,-50%)}
.line-scene .sim-sort-gate:before {content:"";position:absolute;left:5px;top:22px;width:12px;height:12px;border-radius:50%;background:radial-gradient(circle at 35% 30%,#e9f6ff,#5c778c 52%,#152c40);border:2px solid #e4f4ff;box-shadow:0 2px 8px #07111d}
.line-scene .sim-sort-gate .gate-arm {position:absolute;left:15px;top:25px;width:40px;height:7px;border:1px solid #d8e6ed;border-radius:5px;background:linear-gradient(#f2f7f8,#597184);box-shadow:0 2px 6px #07111d;transform:rotate(0);transform-origin:3px 50%;transition:transform .45s cubic-bezier(.2,.8,.25,1),background .3s}
.line-scene .sim-sort-gate small {position:absolute;left:50%;top:-4px;transform:translate(-50%,-100%);padding:4px 6px;border:1px solid #62d9ae88;border-radius:5px;background:#0b302bdc;color:#abf4d7;font:700 7px 'DM Sans',sans-serif;letter-spacing:.07em;white-space:nowrap;box-shadow:0 3px 10px #08121d88}
.ss3d-shell.flag-divert .line-scene .sim-sort-gate .gate-arm {transform:rotate(-52deg);background:linear-gradient(#ffd3a3,#c85245);border-color:#ffe1c4}
.ss3d-shell.flag-divert .line-scene .sim-sort-gate small {border-color:#ff918688;background:#491d23e8;color:#ffd4cf}
.line-scene .route-labels {position:absolute;z-index:11;right:1.5%;top:48%;display:flex;flex-direction:column;gap:7px}.line-scene .route-label {padding:7px 9px;background:#132b42dd;border:1px solid #46647d;border-radius:7px;color:#c9d9e5;font-size:8px;letter-spacing:.06em;backdrop-filter:blur(5px)}.line-scene .route-label.active.main {border-color:#50d8a4;color:#9cf3d0}.line-scene .route-label.active.reject {border-color:#ff8178;color:#ffd4d0;background:#592a32}
.line-scene .data-packets {position:absolute;z-index:10;left:43%;top:35%;display:flex;gap:6px;opacity:0;transition:opacity .2s}.line-scene.ai-active .data-packets {opacity:1}.data-packets i {width:8px;height:8px;border-radius:2px;background:#55e6ff;box-shadow:0 0 12px #39d9ff;animation:data-hop .7s ease-in-out infinite alternate}.data-packets i:nth-child(2){animation-delay:.15s}.data-packets i:nth-child(3){animation-delay:.3s}
@keyframes data-hop {to{transform:translateX(32px);opacity:.35}}
.line-scene .inspection-overlay {position:absolute;z-index:10;left:1.5%;bottom:12px;width:180px;padding:10px;border:1px solid #78a0b9;border-radius:10px;background:#0c1a2bd9;color:#d6ebfa;box-shadow:0 8px 24px #050d17aa;backdrop-filter:blur(7px)}
.line-scene .inspection-overlay b {display:block;font-size:9px;letter-spacing:.07em}.line-scene .inspection-overlay .overlay-score {font-size:24px;font-weight:800;color:#62dcff;margin:5px 0}.ss3d-shell.flag-route .line-scene .overlay-score{color:#ff8580}.line-scene .inspection-overlay small{font-size:8px;color:#a4c0d4}
.line-result-strip {display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin:9px 0 4px;padding:10px 14px;border:1px solid #35536e;border-radius:10px;background:#10243a;color:#c5d8e7;font-size:10px;letter-spacing:.08em}
.line-result-strip .result-cell {display:flex;align-items:center;gap:8px}.line-result-strip b {font-size:13px;color:#e7f4fc;letter-spacing:.03em}.line-result-strip .result-score {font-size:18px;color:#69dcff}.ss3d-shell.flag-route .line-result-strip .result-score,.ss3d-shell.flag-route .line-result-strip .result-decision {color:#ff9690}.line-result-strip .result-destination {color:#96f0c9}.ss3d-shell.flag-route .line-result-strip .result-destination {color:#ffb0a9}
/* Sample position is refreshed from the same elapsed clock as the stage indicator. */
.ss3d-shell.running .line-scene .line-station,.ss3d-shell.running .line-result-strip,.ss3d-shell.running .ss3d-step,.ss3d-shell.running .line-scene .beam {animation:none!important}
.ss3d-shell.paused .line-scene .moving-sample {animation-play-state:paused!important}
.ss3d-shell.running .line-result-strip {opacity:1;transform:none}
.ss3d-shell.running .line-scene .beam {opacity:0}.ss3d-shell.running .line-scene .line-station.light-on .beam {opacity:1}
.scene-live-status {position:absolute;z-index:12;left:15px;top:15px;width:min(300px,calc(100% - 30px));padding:12px 14px;border:1px solid #d6edf477;border-radius:12px;background:linear-gradient(145deg,#081b2cE8,#0b2638dc);color:#edfaff;box-shadow:0 8px 25px #07111d66;backdrop-filter:blur(9px);pointer-events:none}
.scene-live-meta {display:flex;align-items:center;gap:7px;color:#9dbdca;font-size:9px;font-weight:800;letter-spacing:.11em}.scene-live-meta small {margin-left:auto;color:#77d9d1;font-size:9px;letter-spacing:.06em}
.scene-live-dot {width:7px;height:7px;border-radius:50%;background:#34d7a3;box-shadow:0 0 10px #34d7a3}
.scene-live-dot.paused {background:#f0bb59;box-shadow:0 0 10px #f0bb59}
.scene-live-status strong {display:block;margin:6px 0 3px;font-size:13px;letter-spacing:.04em;color:#fff}
.scene-live-status p {margin:0;color:#b8d0db;font-size:10px}
.scene-live-track {display:grid;grid-template-columns:repeat(6,1fr);gap:4px;margin-top:10px}.scene-live-track i {height:3px;border-radius:5px;background:#ffffff25}.scene-live-track i.done,.scene-live-track i.current {background:#45d4bc}.scene-live-track i.current {box-shadow:0 0 8px #45d4bc}
.sim-readout {background:#101f31;border:1px solid #2b465e;border-radius:13px;padding:13px 15px;color:#deebf5;min-height:102px}.sim-readout b {font-size:11px;letter-spacing:.08em;color:#90dfff}.sim-readout p {font-size:11px;color:#b6c9d9;margin:7px 0 0}.sim-band-grid {display:grid;grid-template-columns:repeat(6,1fr);gap:4px;margin:8px 0}.sim-band-grid span {height:22px;border-radius:3px;border:1px solid #7190a733}.sim-progress {background:#12263a;border:1px solid #294761;border-radius:10px;padding:10px 13px;color:#eaf5fd;font-size:13px;font-weight:700;margin-bottom:9px}
@media(prefers-reduced-motion:reduce){.ss3d-belt{animation:none}.ss3d-grain{animation:none!important;left:82%;bottom:73px}}
@media(max-width:700px){.ss3d-scene.line-scene{height:clamp(300px,60vw,420px)}}
div[data-testid="stMetric"] {background:white;border:1px solid #e3eaf2;padding:12px;border-radius:13px}
button[kind="primary"] {background:#1674ba}
/* Global visual refresh: cool, low-glare canvas with crisp teal accents. */
:root {--ss-ink:#172b3d;--ss-muted:#66798b;--ss-line:#dbe5e9;--ss-surface:#ffffff;--ss-canvas:#eef3f3;--ss-teal:#087e78;--ss-teal-dark:#07645f;--ss-cyan:#27b6ad;--ss-shadow:0 10px 28px #183a4210;--primary-color:#087e78}
.stApp {background:radial-gradient(ellipse at 82% 0%,#e4f1ef 0,transparent 34%),linear-gradient(180deg,#f2f6f5 0%,#edf2f3 100%);color:var(--ss-ink)}
[data-testid="stMainBlockContainer"] {padding-top:2rem;padding-bottom:3rem;max-width:1500px}
h1,h2,h3,h4 {color:#183448;letter-spacing:-.025em}
[data-testid="stSidebar"] {background:linear-gradient(165deg,#102b39 0%,#143c49 65%,#0f303d 100%);border-right:1px solid #284c56}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,[data-testid="stSidebar"] label {color:#d9e8ea}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color:#8fb4b8}
[data-testid="stSidebar"] hr {border-color:#ffffff20}
[data-testid="stSidebar"] [role="radiogroup"] {gap:5px}
[data-testid="stSidebar"] [role="radiogroup"] label {padding:9px 12px;border-radius:9px;transition:background .18s ease}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background:#ffffff12}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:#ffffff17;color:#fff}
.hero {background:linear-gradient(112deg,#123243 0%,#14505b 66%,#16766f 100%);border:1px solid #ffffff20;box-shadow:0 16px 36px #12354322;position:relative;overflow:hidden}
.hero:after {content:"";position:absolute;width:300px;height:300px;right:-105px;top:-175px;border:1px solid #ffffff1a;border-radius:50%;box-shadow:0 0 0 28px #ffffff08,0 0 0 62px #ffffff06;pointer-events:none}
.hero h1 {letter-spacing:-.04em}.hero .badge {background:#d7f4ed;color:#08645d}
.card,div[data-testid="stMetric"] {background:#ffffff;border-color:#dce7e8;box-shadow:var(--ss-shadow)}
.kicker {color:#68858d}.metric {color:#123b49}.muted {color:#6a7e88}
.stage {background:#eef6f5;border-color:#d5e7e4;color:#285b5a}
div[data-testid="stMetric"] {padding:15px 17px;border-radius:14px}
div[data-testid="stMetricLabel"] {color:#71868c}
div[data-testid="stMetricValue"] {color:#153d4a}
div[data-testid="stMetricDelta"] svg {color:var(--ss-teal)}
div[data-testid="stButton"] > button {border:1px solid #d3e0e2;border-radius:10px;background:#fff;color:#244250;font-weight:700;transition:transform .16s ease,box-shadow .16s ease,border-color .16s ease}
div[data-testid="stButton"] > button:hover {border-color:#79b8b2;color:#075f5a;box-shadow:0 5px 14px #0b706b1a;transform:translateY(-1px)}
div[data-testid="stButton"] > button[kind="primary"] {background:linear-gradient(110deg,#087e78,#11988c);border-color:#087e78;color:white;box-shadow:0 6px 16px #087e7830}
div[data-testid="stButton"] > button[kind="primary"]:hover {background:linear-gradient(110deg,#076c67,#0b8178);color:white}
[data-baseweb="tab-list"] {gap:6px;border-bottom:1px solid #dbe5e7}
[data-baseweb="tab"] {border-radius:9px 9px 0 0;color:#617883}
[aria-selected="true"][data-baseweb="tab"] {color:#087e78;border-bottom-color:#087e78}
[data-baseweb="select"] > div,[data-baseweb="input"] > div,[data-baseweb="textarea"] textarea {border-color:#d4e0e2;border-radius:9px;background:#fff}
[data-baseweb="select"] > div:focus-within,[data-baseweb="input"] > div:focus-within,[data-baseweb="textarea"] textarea:focus {border-color:#2b9d94;box-shadow:0 0 0 1px #2b9d94}
[data-testid="stFileUploader"] section {border-color:#b9d6d2;background:#f7fbfa;border-radius:12px}
[data-testid="stDataFrame"],[data-testid="stTable"] {border:1px solid #dce6e7;border-radius:12px;overflow:hidden}
[data-testid="stExpander"] {border-color:#dbe5e7;border-radius:12px;background:#ffffffb8}
[data-testid="stAlert"] {border-radius:11px}
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] * {color:#526d78!important}
[data-testid="stAppViewContainer"] [data-testid="stRadio"] label,[data-testid="stAppViewContainer"] [data-testid="stRadio"] label * {color:#355662!important}
[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"],[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] * {color:#355662!important}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,[data-testid="stSidebar"] [data-testid="stRadio"] label,[data-testid="stSidebar"] [data-testid="stRadio"] label * {color:#d9e8ea!important}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {color:#8fb4b8!important}
[data-testid="stAppViewContainer"] input[type="radio"] {accent-color:#087e78}
[data-testid="stAppViewContainer"] [data-testid="stSlider"] [role="slider"] {background:#087e78;border-color:#087e78}
[data-testid="stProgressBar"] > div > div {background:linear-gradient(90deg,#087e78,#46c3ae)}
button[kind="primary"] {background:linear-gradient(110deg,#087e78,#11988c)}
@media(max-width:700px){[data-testid="stMainBlockContainer"]{padding:1rem 1rem 2rem}.hero{padding:22px 20px}.hero h1{font-size:30px}}
</style>
""", unsafe_allow_html=True)

def grain_image(kind="clean", seed=4, size=(640,420)):
    rng=np.random.default_rng(seed); w,h=size
    base=np.zeros((h,w,3),dtype=np.uint8); base[:]=[220,211,184]
    # subtle matte tabletop texture
    noise=rng.normal(0,4,(h,w,1)); base=np.clip(base.astype(float)+noise,0,255).astype(np.uint8)
    im=Image.fromarray(base); d=ImageDraw.Draw(im)
    palette=[(194,154,83),(216,178,104),(171,128,67),(232,204,145),(185,145,89)]
    if kind=="flagged": palette += [(48,105,76),(56,67,61),(125,71,49),(35,43,43)]
    count=230
    for _ in range(count):
        x=int(rng.integers(10,w-10)); y=int(rng.integers(10,h-10)); rx=int(rng.integers(5,12)); ry=int(rng.integers(3,7))
        color=palette[int(rng.integers(len(palette)))]
        d.ellipse((x-rx,y-ry,x+rx,y+ry),fill=color,outline=(150,119,75),width=1)
        d.line((x-rx//2,y,x+rx//2,y),fill=tuple(max(0,c-25) for c in color),width=1)
    if kind=="flagged":
        # Visible illustrative inclusions; not actual contaminants or a validated detection.
        d.ellipse((w*.70,h*.18,w*.84,h*.31),fill=(40,56,47),outline=(25,40,36),width=2)
        d.ellipse((w*.20,h*.65,w*.28,h*.73),fill=(96,54,41),outline=(70,40,32),width=2)
    return im.filter(ImageFilter.GaussianBlur(.25))

def sample_index():
    p=ROOT/'data'/'samples'; p.mkdir(parents=True,exist_ok=True)
    if not any(p.glob('*.png')):
        for k in ('clean','flagged'):
            for i in range(1,6): grain_image(k,i+10).save(p/f'{k}_{i:02}.png')
    return sorted(p.glob('*.png'))

def add_history(record):
    st.session_state.history.insert(0,record); st.session_state.history=st.session_state.history[:80]

if 'history' not in st.session_state: st.session_state.history=[]
if 'sample_seq' not in st.session_state: st.session_state.sample_seq=0
if 'selected_kind' not in st.session_state: st.session_state.selected_kind='clean'
if 'current' not in st.session_state: st.session_state.current=None
if 'sim_running' not in st.session_state: st.session_state.sim_running=False
if 'sim_paused' not in st.session_state: st.session_state.sim_paused=False
if 'sim_stage' not in st.session_state: st.session_state.sim_stage=0
if 'sim_elapsed' not in st.session_state: st.session_state.sim_elapsed=0.0
if 'sim_started_at' not in st.session_state: st.session_state.sim_started_at=0.0
if 'sim_stage_seconds' not in st.session_state: st.session_state.sim_stage_seconds=1.0
if 'sim_cycle' not in st.session_state: st.session_state.sim_cycle=0

st.sidebar.markdown("## ◈ SpectraSafe")
st.sidebar.caption("SIMULATION CONSOLE")
page=st.sidebar.radio("NAVIGATION",["Inspection Console","3D Inspection Line Simulation","Deployment Architecture","About & Limitations"],index=1,label_visibility="collapsed")
st.sidebar.divider()
presentation=st.sidebar.toggle("Presentation / Demo Mode",value=False,key="presentation_mode")
if presentation:
    st.markdown("<style>[data-testid='stSidebar']{display:none!important}[data-testid='stToolbar']{visibility:hidden}</style>",unsafe_allow_html=True)
threshold=st.sidebar.slider("Flag threshold",30,90,60,help="Prototype decision boundary; not scientifically validated.")
st.sidebar.markdown("---\n**SYSTEM STATUS**\n\n🟢 Image input · ready\n\n🟢 Preprocessing · ready\n\n🟢 Pseudo-HSI · software simulation\n\n🟢 AI model · demo model\n\n🟡 PLC interface · simulated\n\n🟡 HSI camera · simulated")
payload=get_model(str(ROOT/'models'/'classifier.pkl'))

def header():
    st.markdown('<div class="hero"><span class="badge">SOFTWARE PROTOTYPE · SIMULATION</span><h1 style="margin-top:10px">SpectraSafe</h1><p>Inline Hyperspectral-AI Food Contamination Screening</p><p class="subline">REAL-TIME • NON-DESTRUCTIVE • INTELLIGENT • AUTOMATED SEGREGATION</p><p class="subline">Concept validation using image-based pseudo-hyperspectral simulation</p></div>',unsafe_allow_html=True)

LINE_STEPS=[
    ("PRODUCT ACQUISITION","Sample enters inspection line"),
    ("CONTROLLED ILLUMINATION","Uniform NIR / visible illumination · simulated"),
    ("HYPERSPECTRAL ACQUISITION","12-band pseudo-HSI scan · software"),
    ("EDGE AI ANALYSIS","Preprocess · spectral/spatial features · ML · risk"),
    ("DECISION","Prototype contamination-risk screening"),
    ("SORTING / SEGREGATION","Simulated actuator routes the sample"),
]

def line_scene_html(result, stage, paused=False, cycle=0, running=False, duration=15.6, elapsed=0.0):
    kind=result['kind']; flag=(result['decision']=='FLAG'); decision_visible=stage>=4
    # Product centers follow the visible roller bed from its near-left entry to
    # the scan point, then along the belt or down the reject chute.
    # These are the stationary fallback locations; the live route below is
    # timed to pass directly beneath the visible line-scan camera in step 3.
    sample_path=[(0,4,56),(16.667,25,50),(30,39,44),(33.333,43,42),(40,49,40),(46,54,39),(50,58,38),(66.667,69,35),(83.333,81,32),(87,84,32),(100,94,30)]
    if flag: sample_path[-2:]=[(93,90,45),(100,96,57)]
    if running:
        # Calculate from the same monotonic clock as the process stage. This
        # avoids CSS animation-delay drift during Streamlit fragment refreshes.
        pct=min(100,max(0,elapsed/max(duration,1e-6)*100))
        for (p0,x0,y0),(p1,x1,y1) in zip(sample_path,sample_path[1:]):
            if pct<=p1:
                f=(pct-p0)/max(p1-p0,1e-6); x=x0+(x1-x0)*f; y=y0+(y1-y0)*f; break
        else: x,y=sample_path[-1][1:]
        left,top=f"{x:.2f}%",f"{y:.2f}%"
    else:
        static_path=[("4%","56%"),("25%","50%"),("49%","40%"),("60%","38%"),("74%","33%"),("94%","30%" if not flag else "57%")]
        left,top=static_path[min(stage,5)]
    transform="translate(-50%,-50%) rotate(-6deg)"
    if stage==5 and flag: transform="translate(-50%,-50%) rotate(12deg)"
    stations=[
        ("PRODUCT ENTRY","IMAGE INPUT","◉"),
        ("SIMULATED NIR","ILLUMINATION","☼"),
        ("PSEUDO-HSI","LINE-SCAN CAMERA","▣"),
        ("EDGE COMPUTER","AI / GPU","AI"),
        ("RISK GATE","PASS / FLAG","◇"),
        ("SORTER","SERVO SIM","⇢"),
    ]
    station_html=[]
    station_xy=[("12%","49%"),("29%","42%"),("50%","24%"),("66%","36%"),("75%","45%"),("86%","53%")]
    for idx,((title,sub,icon),(x,y)) in enumerate(zip(stations,station_xy)):
        cls="ss3d-station line-station"
        if stage==idx: cls+=" hot"
        if idx==1 and stage==1: cls+=" light-on"
        station_html.append(f'<div class="{cls}" title="STEP {idx+1} · {title}" style="left:{x};top:{y}"><span class="line-pin-number">{idx+1}</span><div class="beam"></div></div>')
    risk_text=f"{result['risk']}%" if decision_visible else "CALCULATING"
    decision_text=result['decision'] if decision_visible else "PENDING"
    route_class="flag-route" if flag and stage>=4 else ""
    gate_class="flag-divert" if flag and stage>=5 else ""
    reject_class="reject-cycle" if flag and running else ""
    scan_cls="scan-active" if stage==2 else ""
    ai_cls="ai-active" if stage==3 else ""
    paused_class="paused" if paused else ""
    running_class="running" if running else ""
    timeline_vars=""
    step_title,step_detail=LINE_STEPS[min(stage,5)]
    scene_state="PAUSED" if paused else ("LIVE PROCESS" if running else ("SEQUENCE COMPLETE" if stage==5 else "READY · PRESS START"))
    track_html="".join(f'<i class="{"done" if i<stage else "current" if i==stage else ""}"></i>' for i in range(6))
    scene_svg='<img src="https://raw.githubusercontent.com/anilsenapati964-rgb/spectrasafe-prototype/main/static/industrial_conveyor.jpg" alt="Industrial conveyor with fixed inspection camera, edge computer and sorting gate">'
    return f"""
<div class="ss3d-shell {route_class} {gate_class} {reject_class} {paused_class} {running_class} cycle-{cycle}" style="{timeline_vars}">
 <div class="ss3d-head"><div><strong>INLINE INSPECTION LINE · DIGITAL TWIN VIEW</strong><br><span style="font-size:9px;color:#a9c4dc;letter-spacing:.12em">3D SYSTEM SIMULATION — HARDWARE / ACTUATORS SIMULATED</span></div><span class="ss3d-tag">SAMPLE SS-{result.get('record',{}).get('Sample ID','DEMO').replace('GRAIN-','')}</span></div>
 <div class="ss3d-scene line-scene {scan_cls} {ai_cls}" aria-label="Simulated conveyor digital twin; numbered pins map to the six process steps below">
   <div class="line-backdrop">{scene_svg}</div>
   {''.join(station_html)}
   <div class="scene-live-status"><div class="scene-live-meta"><span class="scene-live-dot {"paused" if paused else ""}"></span>{scene_state}<small>STEP {stage+1} / 6</small></div><strong>{step_title}</strong><p>{step_detail}</p><div class="scene-live-track">{track_html}</div></div>
   <div class="moving-sample" style="left:{left};top:{top};transform:{transform}"></div>
   <div class="sim-sort-gate" aria-label="Automatic sorting gate: {"diverting flagged sample" if flag and stage>=5 else "open for passing sample"}"><span class="gate-arm"></span><small>{"GATE · DIVERT" if flag and stage>=5 else "GATE · PASS"}</small></div>
   <div class="scan-line" style="left:50%;top:40%;height:54px"></div>
   <div class="data-packets" style="left:57%;top:39%"><i></i><i></i><i></i></div>
 </div>
 <div class="line-result-strip" aria-label="Inspection result and sample route">
  <span class="result-cell">RISK <b class="result-score">{risk_text}</b></span>
  <span class="result-cell">DECISION <b class="result-decision">{decision_text}</b></span>
  <span class="result-cell">ROUTE <b class="result-destination">{"QUARANTINE / INSPECTION" if flag else "MAIN PRODUCTION LINE"}</b></span>
  <span style="color:#819bb0;font-size:8px">SIMULATED · NOT VALIDATED</span>
 </div>
 <div class="ss3d-steps">{''.join(f'<div class="ss3d-step {"active" if i==stage else ""}"><em>STEP {i+1}/6</em>{step[0]}</div>' for i,step in enumerate(LINE_STEPS))}</div>
</div>
"""

def analyze_image(image, kind):
    model=payload['model']; raw,den,norm,mask=preprocess(image); cube=pseudo_cube(np.asarray(norm)); features=extract_features(cube,mask)
    proba=float(model.predict_proba(features.reshape(1,-1))[0][1])
    risk=score_risk(proba,kind); decision='FLAG' if risk>=threshold else 'PASS'; route=route_for(decision)
    return {'image':image,'raw':raw,'den':den,'norm':norm,'mask':mask,'cube':cube,'features':features,'signature':signature(cube,mask),'risk':risk,'decision':decision,'route':route,'proba':proba,'kind':kind}

def inspect(image, kind, demo=False):
    result=analyze_image(image,kind)
    st.session_state.sample_seq+=1
    rec={'Sample ID':f'GRAIN-{st.session_state.sample_seq:04d}','Risk Score':result['risk'],'Decision':result['decision'],'Action':result['route']['action'],'Time':datetime.now().strftime('%H:%M:%S')}
    add_history(rec)
    result['record']=rec
    st.session_state.current=result
    return result

if page=="Deployment Architecture":
    header(); st.subheader("Future Production System")
    st.caption("FUTURE HARDWARE DEPLOYMENT ARCHITECTURE · NO HARDWARE IS CONNECTED IN THIS PROTOTYPE")
    cols=st.columns(6)
    for c,title,body in zip(cols,["Line-scan HSI","NIR / visible","Edge GPU / NPU","AI model","PLC / controller","Air jet / servo"],["Future camera","Controlled lighting","On-device inference","Risk classification","Object tracking","Reject / quarantine"]):
        with c: st.markdown(f'<div class="card" style="text-align:center;min-height:135px"><div style="font-size:25px">◈</div><b>{title}</b><p class="muted">{body}</p></div>',unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;font-size:24px;color:#1785c4'>↓　↓　↓　↓　↓</div>",unsafe_allow_html=True)
    st.info("This prototype demonstrates the proposed software workflow using ordinary images and derived pseudo-spectral bands. Future camera, illumination, edge compute, controller, and actuator integration would require engineering and validation.")
    st.markdown("<h3 style='text-align:center;margin-top:42px'>SpectraSafe</h3><p style='text-align:center;color:#647b94'>Towards safer food, healthier people and stronger food systems.</p>",unsafe_allow_html=True)
elif page=="3D Inspection Line Simulation":
    sample_index()
    st.markdown("""
<div class="line-status"><small>SPECTRASAFE · INLINE HYPERSPECTRAL INSPECTION</small><h2>3D INSPECTION LINE SIMULATION</h2>
<div class="line-status-chips"><span>SYSTEM STATUS: SIMULATION</span><span>CAMERA: SIMULATED</span><span>ILLUMINATION: SIMULATED</span><span>EDGE AI: SOFTWARE ACTIVE</span><span>PLC: SIMULATED</span><span>SORTER: SIMULATED</span></div></div>
""",unsafe_allow_html=True)
    st.caption("Pseudo-isometric digital-twin view · browser animation only · no physical equipment is connected.")
    case_col,mode_col=st.columns([1,2])
    with case_col: case=st.radio("SELECT DEMO PRODUCT",["CLEAN SAMPLE","FLAGGED SAMPLE"],horizontal=True,key="sim_case")
    kind="clean" if case=="CLEAN SAMPLE" else "flagged"
    image=Image.open(ROOT/'data'/'samples'/f'{kind}_01.png').convert('RGB')
    with mode_col: st.markdown(f'<div style="color:#496772;font-size:13px;padding:7px 0">Selected product: <strong style="color:#183f4a">{kind.upper()}</strong> · existing prototype score and {threshold}% flag threshold.</div>',unsafe_allow_html=True)
    controls=st.columns(4)
    with controls[0]: start_clicked=st.button("▶ START INSPECTION",width="stretch",disabled=st.session_state.sim_running,key="line_start")
    with controls[1]: pause_clicked=st.button("▶ RESUME" if st.session_state.sim_paused else "⏸ PAUSE",width="stretch",disabled=not st.session_state.sim_running,key="line_pause")
    with controls[2]: restart_clicked=st.button("↻ RESTART",width="stretch",key="line_restart")
    with controls[3]: auto_clicked=st.button("⚡ AUTO DEMO MODE",type="primary",width="stretch",disabled=st.session_state.sim_running,key="line_auto")
    if start_clicked or auto_clicked:
        result=inspect(image,kind,demo=True)
        st.session_state.sim_result=result; st.session_state.sim_running=True; st.session_state.sim_paused=False; st.session_state.sim_stage=0; st.session_state.sim_elapsed=0.0
        # A slightly longer stage interval softens the sample speed while
        # keeping the camera pass and every process label on the same clock.
        st.session_state.sim_started_at=time.monotonic(); st.session_state.sim_stage_seconds=1.1 if auto_clicked else 2.9; st.session_state.sim_cycle+=1
    elif pause_clicked:
        if st.session_state.sim_paused:
            st.session_state.sim_started_at=time.monotonic(); st.session_state.sim_paused=False
        else:
            st.session_state.sim_elapsed+=max(0,time.monotonic()-st.session_state.sim_started_at); st.session_state.sim_paused=True
            st.session_state.sim_stage=min(5,int(st.session_state.sim_elapsed/st.session_state.sim_stage_seconds))
    elif restart_clicked:
        st.session_state.sim_running=False; st.session_state.sim_paused=False; st.session_state.sim_stage=0; st.session_state.sim_elapsed=0.0; st.session_state.sim_cycle+=1
    if start_clicked or auto_clicked or pause_clicked or restart_clicked:
        st.rerun()
    result=st.session_state.get('sim_result')
    if result is None or result.get('kind')!=kind:
        result=analyze_image(image,kind); st.session_state.sim_result=result
        st.session_state.sim_running=False; st.session_state.sim_paused=False; st.session_state.sim_stage=0; st.session_state.sim_elapsed=0.0
    band_colors=["#5843a4","#3a6ed0","#228bb2","#36a984","#a2ae43","#d0a037","#da793b","#c7514d","#a54365","#823c83","#5d4a9e","#5795c3"]
    bands_html="".join(f'<span title="Band {i+1}" style="background:linear-gradient(120deg,{color},#192f49)"></span>' for i,color in enumerate(band_colors))
    vals=np.asarray(result['signature'],dtype=float); low,high=float(vals.min()),float(vals.max()); spread=max(high-low,1e-5)
    spectrum_points=" ".join(f'{i*100/(len(vals)-1):.1f},{38-(float(v)-low)/spread*30:.1f}' for i,v in enumerate(vals))
    @st.fragment(run_every=.12 if st.session_state.sim_running and not st.session_state.sim_paused else None)
    def live_inspection_line():
        elapsed=st.session_state.sim_elapsed
        if st.session_state.sim_running and not st.session_state.sim_paused:
            elapsed+=max(0,time.monotonic()-st.session_state.sim_started_at)
            stage=min(5,int(elapsed/st.session_state.sim_stage_seconds)); st.session_state.sim_stage=stage
            finished=elapsed>=st.session_state.sim_stage_seconds*len(LINE_STEPS)
            if finished:
                st.session_state.sim_running=False; st.session_state.sim_paused=False; stage=5; st.session_state.sim_stage=5
                st.session_state.sim_elapsed=st.session_state.sim_stage_seconds*len(LINE_STEPS)
        else: stage=int(st.session_state.sim_stage); finished=False
        step_name,step_detail=LINE_STEPS[stage]
        state_label="PAUSED" if st.session_state.sim_paused else ("SEQUENCE COMPLETE" if not st.session_state.sim_running and stage==5 else ("INSPECTION IN PROGRESS" if st.session_state.sim_running else "READY · PRESS START"))
        st.markdown(line_scene_html(result,stage,st.session_state.sim_paused,st.session_state.sim_cycle,
                                   st.session_state.sim_running,st.session_state.sim_stage_seconds*len(LINE_STEPS),elapsed),unsafe_allow_html=True)
        st.markdown(f'<div class="sim-progress">STEP {stage+1} / 6　·　{step_name}<span style="float:right;color:#8edfff">{state_label}</span><div style="font-size:10px;color:#a9c0d3;font-weight:400;margin-top:5px">{step_detail}</div></div>',unsafe_allow_html=True)
        st.progress((stage+1)/len(LINE_STEPS),text=f"STEP {stage+1} / 6 · {step_name}")
        rec=result.get('record',{}); sample_id='SS-'+rec.get('Sample ID','DEMO').replace('GRAIN-',''); visible=stage>=4
        cards=st.columns(5)
        values=[("SAMPLE ID",sample_id),("INSPECTION MODE","AUTO DEMO" if st.session_state.sim_stage_seconds<1 else "DEMO"),("CURRENT STAGE",f"{stage+1}/6 · {step_name}"),("RISK SCORE",f"{result['risk']}%" if visible else "ANALYZING"),("DECISION",result['decision'] if visible else "PENDING")]
        for col,(label,value) in zip(cards,values):
            with col: st.markdown(f'<div class="sim-readout"><b>{label}</b><p>{value}</p></div>',unsafe_allow_html=True)
        st.markdown("#### Camera data → Edge AI")
        vis_col,ai_col=st.columns([1.05,.95])
        with vis_col:
            st.markdown('<div class="sim-readout"><b>PSEUDO-HSI VISUALIZATION — SOFTWARE SIMULATION · 12 SIMULATED BANDS</b>',unsafe_allow_html=True)
            st.markdown(f'<div class="sim-band-grid">{bands_html}</div>',unsafe_allow_html=True)
            st.markdown(f'<svg viewBox="0 0 100 44" style="width:100%;height:62px"><polyline points="{spectrum_points}" fill="none" stroke="#6fe3ff" stroke-width="1.8"/><line x1="0" y1="39" x2="100" y2="39" stroke="#45637b" stroke-width=".6"/></svg><small style="color:#9bb5c9">12-band line-scan simulation · no measured wavelength/reflectance</small></div>',unsafe_allow_html=True)
        with ai_col:
            checks=[("Preprocessing",stage>=3),("Spectral features",stage>=3),("Spatial features",stage>=3),("ML classification",stage>=3),("Risk scoring",stage>=4)]
            lines="".join(f'<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #29445b"><span>{name}</span><b style="color:{"#64e2b3" if done else "#7890a5"}">{"✓ COMPLETE" if done else ("● ACTIVE" if stage==3 else "QUEUED")}</b></div>' for name,done in checks)
            st.markdown(f'<div class="sim-readout"><b>STEP 4 — EDGE AI ANALYSIS</b><div style="font-size:10px;color:#c1d0dc;margin-top:6px">{lines}</div><p style="color:#8bdfff">DATA LINK: CAMERA → EDGE AI · {"PACKETS FLOWING" if stage==3 else "SIMULATED"}</p></div>',unsafe_allow_html=True)
        st.caption("SIMULATED SORTING ACTUATOR · camera, illumination, conveyor, PLC and sorter are software representations only.")
        if finished: st.rerun()
    live_inspection_line()
    st.caption("Prototype demonstration — simulated HSI acquisition, AI screening and automated segregation. Not laboratory-validated diagnostic results.")
elif page=="About & Limitations":
    header(); st.subheader("About this prototype")
    st.write("SpectraSafe is an image-based software simulation of a proposed inline food and grain contamination-risk screening workflow. The displayed multi-band cube is mathematically derived from RGB values; no spectral sensor measurements are present.")
    st.markdown("#### Interpretation")
    st.write("Clean and flagged canned demonstrations use deterministic prototype scores (18% and 84%) so the two routing scenarios are repeatable. Uploaded images use a small model trained on procedural demo features. These scores are concept demonstration outputs, not validated probabilities.")
    st.markdown("#### Current limitations")
    st.write("No hyperspectral camera, NIR illuminator, PLC, conveyor, or air jet is connected. No laboratory-confirmed training data are supplied. The demo classifier and metrics must not be used for food safety decisions, pathogen identification, or production release.")
    st.markdown("#### Model snapshot")
    m=payload['metrics']; st.caption(f"Demo Dataset Performance — accuracy {m['accuracy']:.0%} · precision {m['precision']:.0%} · recall {m['recall']:.0%}. Computed on a held-out split of synthetic demo data.")
else:
    header()
    if presentation:
        if st.button("EXIT PRESENTATION MODE",key="exit_presentation"):
            st.session_state.presentation_mode=False
            st.rerun()
    if not presentation:
        st.markdown("### Inspection Console")
        tabs=st.tabs(["Sample Image","Upload Image","Webcam (optional)"])
        image=None; kind="uploaded"
        with tabs[0]:
            samples=sample_index(); options=["CLEAN SAMPLE","FLAGGED SAMPLE","RANDOM SAMPLE"]
            choice=st.radio("Choose demo input",options,horizontal=True,label_visibility="collapsed")
            if choice=="CLEAN SAMPLE": kind="clean"; image=Image.open(ROOT/'data'/'samples'/'clean_01.png').convert('RGB')
            elif choice=="FLAGGED SAMPLE": kind="flagged"; image=Image.open(ROOT/'data'/'samples'/'flagged_01.png').convert('RGB')
            else:
                picked=samples[int(time.time())%len(samples)]; kind='flagged' if 'flagged' in picked.name else 'clean'; image=Image.open(picked).convert('RGB')
            st.caption(f"Selected: {kind.upper()} · procedurally generated grain image · illustrative demo input")
        with tabs[1]:
            up=st.file_uploader("Upload an inspection image",type=["png","jpg","jpeg","webp"])
            if up:
                try: image=Image.open(BytesIO(up.getvalue())).convert('RGB'); kind='uploaded'
                except Exception: st.error("Could not read that image. Please choose a valid PNG or JPEG.")
        with tabs[2]:
            cam=st.camera_input("Capture a sample image")
            if cam:
                try: image=Image.open(BytesIO(cam.getvalue())).convert('RGB'); kind='uploaded'
                except Exception: st.warning("Camera capture is unavailable in this browser. Upload an image instead.")
        colbtn1,colbtn2=st.columns([1,3])
        with colbtn1: run=st.button("▶  DEMO MODE · RUN PIPELINE",type="primary",width="stretch")
        with colbtn2: st.caption("Runs acquisition → preprocessing → pseudo-HSI → features → AI → risk → virtual segregation")
    else:
        image=st.session_state.current['image'] if st.session_state.current else grain_image(st.session_state.selected_kind,18 if st.session_state.selected_kind=='clean' else 29)
        kind=st.session_state.selected_kind
        ca,cb,cc=st.columns([1,1,2])
        with ca:
            if st.button("CLEAN SAMPLE",width="stretch"): st.session_state.selected_kind='clean'; image=grain_image('clean',18); kind='clean'
        with cb:
            if st.button("FLAGGED SAMPLE",width="stretch"): st.session_state.selected_kind='flagged'; image=grain_image('flagged',29); kind='flagged'
        with cc: run=st.button("▶ RUN DEMO PIPELINE",type="primary",width="stretch")
    if run:
        if image is None: st.warning("Choose or upload an image before running the inspection.")
        else:
            bar=st.progress(0,text="Starting software simulation…")
            stages=["Sample acquisition","Pre-processing","Pseudo-HSI generation","Feature extraction","AI classification","Prototype risk score","PASS / FLAG decision","Virtual segregation"]
            for i,s in enumerate(stages):
                bar.progress(int((i+1)*100/len(stages)),text=f"STEP {i+1}/8 · {s}"); time.sleep(.16)
            inspect(image,kind,demo=True); bar.empty()
    current=st.session_state.current
    if current:
        st.markdown("---")
        left,right=st.columns([1.18,.82],gap="large")
        with left:
            st.markdown('<div class="card"><div class="kicker">Live inspection · software input</div>',unsafe_allow_html=True)
            a,b=st.columns(2)
            with a: st.image(current['image'],caption="Input image · software sample",width="stretch")
            with b: st.image(false_color(current['cube']),caption="Pseudo-HSI false color · simulated bands",width="stretch")
            st.markdown("</div>",unsafe_allow_html=True)
            st.markdown('<div class="card"><div class="kicker">Pre-processing</div>',unsafe_allow_html=True)
            p1,p2,p3=st.columns(3)
            for c,img,label in zip([p1,p2,p3],[current['raw'],current['den'],current['norm']],['RAW IMAGE','DENOISED','NORMALIZED + ROI']):
                with c: st.image(img,caption=label,width="stretch")
            st.markdown("</div>",unsafe_allow_html=True)
        with right:
            st.markdown('<div class="card"><div class="kicker">Contamination analysis · prototype screening</div>',unsafe_allow_html=True)
            st.markdown(f"<div class='metric'>{current['risk']}<span style='font-size:19px'>%</span></div><div class='muted'>Prototype Risk Score · {risk_band(current['risk'])} RISK</div>",unsafe_allow_html=True)
            st.progress(current['risk']/100)
            css='pass' if current['decision']=='PASS' else 'flag'; st.markdown(f"<div class='decision {css}'>{'✓' if css=='pass' else '⚠'} {current['decision']}</div>",unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;font-weight:700;margin:10px'>{'CONTINUE PROCESSING' if css=='pass' else 'ACTIVATE SEGREGATION'}</p>",unsafe_allow_html=True)
            st.markdown(f"<div class='muted'>Pattern: {'Baseline-like visual pattern' if css=='pass' else 'Potential contamination pattern'}<br>Model confidence: {max(current['proba'],1-current['proba']):.0%} (demo model output)<br>Sample ID: {current['record']['Sample ID']} · {current['record']['Time']}</div></div>",unsafe_allow_html=True)
        st.markdown(line_scene_html(current,5,False,st.session_state.sample_seq),unsafe_allow_html=True)
        st.caption(f"{current['route']['actuator']} · Animation is an illustrative software simulation; no camera, conveyor, PLC, air-jet, or servo is connected.")
        # Expandable spectral diagnostics keep the recording canvas clean.
        with st.expander("Pseudo-HSI data cube · 12 simulated bands · expand for analysis"):
            st.caption("Pseudo-HSI visualization — software simulation. Bands are derived from RGB image values and are not measured reflectance.")
            for offset in (0,6):
                bandcols=st.columns(6)
                for j,c in enumerate(bandcols):
                    i=offset+j; band=current['cube'][...,i]
                    with c: st.image(band,caption=f"{WAVELENGTHS[i]} nm*",clamp=True,width="stretch")
            st.caption("*Illustrative simulated band labels only; not calibrated wavelengths.")
            fig,ax=plt.subplots(figsize=(8,2.5)); ax.plot(WAVELENGTHS,current['signature'],color='#1674ba',lw=2.5,label='Current sample')
            ax.fill_between(WAVELENGTHS,current['signature'],alpha=.12,color='#1674ba'); ax.set(xlabel='Simulated band label (nm*)',ylabel='Normalized intensity',title='Pseudo-spectral signature'); ax.grid(alpha=.2); ax.legend(frameon=False); st.pyplot(fig); plt.close(fig)
        st.markdown("#### Inspection history")
        rows=st.session_state.history; total=len(rows); passed=sum(r['Decision']=='PASS' for r in rows); flagged=total-passed
        k1,k2,k3,k4=st.columns(4)
        for c,label,value in zip([k1,k2,k3,k4],["TOTAL INSPECTED","PASS","FLAG","QUARANTINED"],[total,passed,flagged,flagged]): c.metric(label,value)
        rate_a,rate_b=st.columns(2)
        with rate_a: st.metric("PASS RATE",f"{passed/total:.0%}" if total else "—")
        with rate_b: st.metric("FLAG RATE",f"{flagged/total:.0%}" if total else "—")
        if rows:
            chart=pd.DataFrame(list(reversed(rows[:20]))); fig,ax=plt.subplots(figsize=(8,2.8)); ax.plot(range(1,len(chart)+1),chart['Risk Score'],marker='o',color='#1674ba',lw=2); ax.axhline(threshold,color='#d64e4e',ls='--',label='Flag threshold'); ax.set_ylim(0,100); ax.set_xlabel('Recent sample sequence'); ax.set_ylabel('Prototype risk score'); ax.grid(alpha=.2); ax.legend(frameon=False); st.pyplot(fig); plt.close(fig)
            st.dataframe(pd.DataFrame(rows),hide_index=True,width="stretch")
        else: st.caption("No inspections recorded yet.")
        m=payload['metrics']; st.caption(f"Demo Dataset Performance · accuracy {m['accuracy']:.0%} · precision {m['precision']:.0%} · recall {m['recall']:.0%} · synthetic training/holdout data")
    else:
        st.info("Choose a clean or flagged sample and run Demo Mode to see the full simulated inspection and routing flow.")
        st.markdown("### Proposed process flow")
        st.markdown(''.join(f"<span class='stage'>{x}</span> → " for x in ["Image acquisition","Pseudo-HSI","Pre-processing","Feature extraction","AI classification","Risk score","PASS / FLAG","Virtual segregation"]),unsafe_allow_html=True)
st.markdown('<div class="foot">Prototype demonstration. Pseudo-hyperspectral processing and risk scoring are software simulations for concept validation and are not laboratory diagnostic results.</div>',unsafe_allow_html=True)
