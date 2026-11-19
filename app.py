import os, uuid, sqlite3, tempfile, threading, smtplib, ssl
from email.message import EmailMessage
from collections import Counter
from functools import wraps

import cv2
from flask import (Flask, request, redirect, url_for, session, g, flash,
                   render_template_string, send_from_directory, jsonify)
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature

from src.predict_pada import predict_pada
from src.analyze import analyze

app = Flask(__name__)
app.secret_key = os.environ.get("NATYASENSE_SECRET_KEY", "change-this-later-use-a-long-random-secret")
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024
# Configure these environment variables to enable password-reset emails.
MAIL_SERVER = os.environ.get("MAIL_SERVER", "smtp.gmail.com")
MAIL_PORT = int(os.environ.get("MAIL_PORT", "587"))
MAIL_USERNAME = os.environ.get("MAIL_USERNAME", "")
MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD", "")
MAIL_FROM = os.environ.get("MAIL_FROM", MAIL_USERNAME)
RESET_TOKEN_MAX_AGE = 30 * 60  # 30 minutes
UPLOAD_DIR, DB = "uploads", "natyasense.db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
LOCK = threading.Lock()          # models run one request at a time
IMG_EXT = {".jpg", ".jpeg", ".png"}
VID_EXT = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
MAX_SECONDS, MAX_FRAMES = 30, 20

# Pada Bheda display names
PADA_DISPLAY = {
    "anchitha": "Anchitha",
    "hayatham": "Hayatham",
    "kunchita": "Kunchita",
    "samapada": "Samapada",
    "ekapada": "Ekapada",
}

# Pada Bheda correction tips (verify these with your guru)
TIPS = {
    "anchitha": "Keep the heels on the floor and lift the toes upward.",
    "hayatham": "Add your guru's description for Hayatham here.",
    "kunchita": "Press the toes on the floor and raise the heels.",
    "samapada": "Keep both feet flat on the floor, together, with weight even.",
    "ekapada": "Keep your weight on one foot and lift the other foot.",
}

TITLES = {
    "mudra": "Mudra",
    "pose": "Pose",
    "padabheda": "Pada Bheda",
}

def canonical_pada(name):
    """Convert Pada Bheda labels into a consistent lowercase format."""
    value = (name or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "anchita": "anchitha",
        "hayata": "hayatham",
    }
    return aliases.get(value, value)
# ---------------------------------------------------------------- password reset email
def send_password_reset_email(recipient, reset_url):
    """Send a time-limited password reset link using configured SMTP credentials."""
    if not MAIL_USERNAME or not MAIL_PASSWORD or not MAIL_FROM:
        raise RuntimeError("Email is not configured. Set MAIL_USERNAME, MAIL_PASSWORD and MAIL_FROM.")
    message = EmailMessage()
    message["Subject"] = "Reset your NatyaSense password"
    message["From"] = MAIL_FROM
    message["To"] = recipient
    message.set_content(
        "We received a request to reset your NatyaSense password.\n\n"
        f"Open this link within 30 minutes to choose a new password:\n{reset_url}\n\n"
        "If you did not request this, you can ignore this email."
    )
    context = ssl.create_default_context()
    with smtplib.SMTP(MAIL_SERVER, MAIL_PORT, timeout=20) as server:
        server.ehlo()
        server.starttls(context=context)
        server.ehlo()
        server.login(MAIL_USERNAME, MAIL_PASSWORD)
        server.send_message(message)


# ---------------------------------------------------------------- database
def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e):
    d = g.pop("db", None)
    if d:
        d.close()

with sqlite3.connect(DB) as c:
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, email TEXT UNIQUE, password TEXT);
    CREATE TABLE IF NOT EXISTS results(id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, filename TEXT, target TEXT, predicted TEXT,
        confidence REAL, feedback TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP);
    """)
    columns = [row[1] for row in c.execute("PRAGMA table_info(results)")]
    if "task" not in columns:
        c.execute("ALTER TABLE results ADD COLUMN task TEXT DEFAULT 'mudra'")

def login_required(f):
    @wraps(f)
    def w(*a, **k):
        if "uid" not in session:
            return redirect(url_for("login"))
        return f(*a, **k)
    return w

# ---------------------------------------------------------------- analysis
def pada_feedback(target, pred, conf):
    target_key = canonical_pada(target)
    pred_key = canonical_pada(pred)

    target_name = PADA_DISPLAY.get(target_key, target or "your selected position")
    pred_name = PADA_DISPLAY.get(pred_key, pred or "Unknown")

    if conf < 0.5:
        return (
            "Not sure about this one. Retake it with both feet clearly "
            "visible, good light, and feet in the lower part of the frame."
        )

    pred_tip = TIPS.get(
        pred_key,
        "Please check the correct foot position with your guru."
    )

    target_tip = TIPS.get(
        target_key,
        "Please check the correct foot position with your guru."
    )

    if target_key and target_key == pred_key:
        return (
            f"Detected {pred_name}, matching your selected practice "
            f"{target_name}. {pred_tip}"
        )

    if target_key:
        return (
            f"You selected {target_name}, but the model detected "
            f"{pred_name}. To correct it, review the selected foot "
            f"position: {target_tip}"
        )

    return f"Detected {pred_name}. {pred_tip}"
def tier_of(score):
    if score is None:
        return None
    return "excellent" if score >= 85 else "good" if score >= 60 else "needs work"

def evaluate(task, path, target):
    """One photo -> dict(label, conf, score, tier, msgs)."""
    with LOCK:
        if task == "padabheda":
            pred, conf = predict_pada(path)
            pred, conf = str(pred), float(conf)
            return {"label": pred, "conf": conf, "score": None, "tier": None,
                    "msgs": [pada_feedback(target, pred, conf)]}
        r = analyze(path, task)
    st = r.get("status")
    if st == "no_detection":
        return {"label": "none", "conf": 0.0, "score": None, "tier": None,
                "msgs": r["messages"]}
    d = {"label": r["label"], "conf": r["confidence"], "score": r.get("score"),
         "tier": r.get("tier"), "msgs": r["messages"]}
    if st == "no_reference":
        d["msgs"] = ["Detected " + r["label"] + ". No reference yet for corrections."]
    return d

def combine(results):
    """Many frames -> one result (majority label, average score)."""
    good = [r for r in results if r["label"] != "none"]
    if not good:
        return results[0] if results else {"label": "none", "conf": 0.0,
                "score": None, "tier": None, "msgs": ["Nothing could be read from this video."]}
    top = Counter(r["label"] for r in good).most_common(1)[0][0]
    same = [r for r in good if r["label"] == top]
    scores = [r["score"] for r in same if r["score"] is not None]
    score = round(sum(scores) / len(scores), 1) if scores else None
    msgs = [m for m, _ in Counter(m for r in same for m in r["msgs"]).most_common(3)]
    return {"label": top, "conf": sum(r["conf"] for r in same) / len(same),
            "score": score, "tier": tier_of(score), "msgs": msgs}

def run_video(task, path, target):
    """Sample up to 20 frames from a video of at most 30 s."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if n <= 0:
        cap.release()
        return None, None, "Could not read this video."
    if n / fps > MAX_SECONDS + 1:
        cap.release()
        return None, None, "Video is longer than 30 seconds. Please trim it."
    idxs = sorted(set(int(i * n / MAX_FRAMES) for i in range(MAX_FRAMES))) if n > MAX_FRAMES else range(n)
    results, thumb = [], None
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        if thumb is None:
            thumb = uuid.uuid4().hex + ".jpg"
            cv2.imwrite(os.path.join(UPLOAD_DIR, thumb), frame)
        tmp = os.path.join(UPLOAD_DIR, "tmp_" + uuid.uuid4().hex + ".jpg")
        cv2.imwrite(tmp, frame)
        try:
            results.append(evaluate(task, tmp, target))
        finally:
            os.remove(tmp)
    cap.release()
    if not results:
        return None, None, "Could not read frames from this video."
    return combine(results), thumb, None

def feedback_for_result(task, d):
    """Return consistent confidence-based feedback for all tasks."""

    title = TITLES[task]
    label = d.get("label", "none")
    display_label = label.replace("_", " ").title()

    messages = d.get("msgs") or []
    correction = " ".join(
        str(m).strip() for m in messages if str(m).strip()
    )

    # Case 1: No movement or body part detected
    if label == "none":
        return correction or (
            f"The {title} could not be identified clearly. "
            "Please try again with good lighting and keep the movement visible."
        )

    # Case 2: Get confidence or score
    if task == "padabheda":
        confidence = d.get("conf")

        if confidence is None:
            confidence = d.get("confidence")

        if confidence is None:
            confidence = d.get("score")
    else:
        confidence = d.get("score")

        if confidence is None:
            confidence = d.get("conf")

        if confidence is None:
            confidence = d.get("confidence")


    # Case 3: Convert confidence into percentage
    if confidence is not None:
        confidence = float(confidence)

        if confidence <= 1:
            confidence *= 100

        # Correct performance
        if confidence > 85:
            return (
                f"Your performed {title} ({display_label}) "
                "is correct."
            )

    # Case 4: Show correction feedback
    if correction:
        return correction

    # Case 5: General fallback
    return (
        f"Your {title} ({display_label}) needs correction. "
        "Please compare your movement with the reference and practise again."
    )

def to_text(task, d, note=""):
    """Store the detected name and correction/success feedback, without scores."""
    return feedback_for_result(task, d)

# ---------------------------------------------------------------- page theme
BASE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NatyaSense</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=Mukta:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{--maroon:#4a0816;--kumkum:#b3122e;--zari:#d4a017;--ivory:#fff6e3;--peacock:#0f6b6b;--ink:#2b1a14}
*{box-sizing:border-box}
html{background:var(--maroon)}
body{margin:0;min-height:100vh;font-family:Mukta,Arial,sans-serif;color:var(--ink);line-height:1.55;
background:
 url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 200'%3E%3Cg fill='none' stroke='%23d4a017' stroke-opacity='.35' stroke-width='1'%3E%3Ccircle cx='100' cy='100' r='96'/%3E%3Ccircle cx='100' cy='100' r='72'/%3E%3Ccircle cx='100' cy='100' r='44'/%3E%3Cellipse cx='100' cy='46' rx='11' ry='28'/%3E%3Cellipse cx='100' cy='46' rx='11' ry='28' transform='rotate(45 100 100)'/%3E%3Cellipse cx='100' cy='46' rx='11' ry='28' transform='rotate(90 100 100)'/%3E%3Cellipse cx='100' cy='46' rx='11' ry='28' transform='rotate(135 100 100)'/%3E%3Cellipse cx='100' cy='46' rx='11' ry='28' transform='rotate(180 100 100)'/%3E%3Cellipse cx='100' cy='46' rx='11' ry='28' transform='rotate(225 100 100)'/%3E%3Cellipse cx='100' cy='46' rx='11' ry='28' transform='rotate(270 100 100)'/%3E%3Cellipse cx='100' cy='46' rx='11' ry='28' transform='rotate(315 100 100)'/%3E%3C/g%3E%3C/svg%3E") no-repeat right -120px top 90px/460px,
 radial-gradient(rgba(212,160,23,.28) 1.6px,transparent 2.2px) 0 0/30px 30px,
 radial-gradient(rgba(212,160,23,.28) 1.6px,transparent 2.2px) 15px 15px/30px 30px,
 radial-gradient(circle at 50% -10%,#8c1230 0,var(--maroon) 62%,#25040b 100%);
background-attachment:fixed}
.zari{height:16px;background-color:var(--maroon);
 background-image:linear-gradient(135deg,var(--zari) 25%,transparent 25%),linear-gradient(225deg,var(--zari) 25%,transparent 25%),linear-gradient(315deg,var(--zari) 25%,transparent 25%),linear-gradient(45deg,var(--zari) 25%,transparent 25%);
 background-size:16px 16px;background-position:-8px 0,-8px 0,0 0,0 0;border-bottom:2px solid var(--zari)}
header{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:14px 18px;max-width:760px;margin:auto}
.logo{font-family:'Cormorant Garamond',Georgia,serif;font-size:32px;font-weight:700;color:var(--zari);letter-spacing:1px;text-decoration:none}
.logo small{display:block;font-family:Mukta,Arial;font-size:13px;color:#f3d98b;letter-spacing:0;font-weight:400}
nav a{color:var(--ivory);margin-left:14px;text-decoration:none;border-bottom:1px solid var(--zari)}
main{max-width:760px;margin:0 auto;padding:0 14px 40px}
.panel{background:var(--ivory);border:2px solid var(--zari);outline:1px solid var(--zari);outline-offset:4px;
 border-radius:4px 28px 4px 28px;padding:20px;margin:14px 0 22px;box-shadow:0 10px 30px rgba(0,0,0,.45)}
h2,h3{font-family:'Cormorant Garamond',Georgia,serif;color:var(--kumkum);margin:.2em 0 .5em;font-size:28px}
h4{margin:.8em 0 .3em;color:var(--peacock)}
label {
    display: block;
    font-weight: 700;
    margin-top: 10px;
    color: #1a0808 !important;
    opacity: 1 !important;
    font-size: 17px !important;
}
input,select{width:100%;padding:11px;margin:5px 0;border:1.5px solid #c9a24a;border-radius:8px;background:#fffdf7;font:inherit}
input:focus,select:focus,button:focus-visible,.tab:focus-visible{outline:3px solid var(--peacock);outline-offset:2px}
button,.btn{display:inline-block;width:100%;padding:12px;margin:8px 0;background:linear-gradient(var(--kumkum),#8c0d22);
 color:#fff;border:1.5px solid var(--zari);border-radius:999px;font:600 17px Mukta,Arial;cursor:pointer;text-align:center;text-decoration:none}
button.alt{background:linear-gradient(var(--peacock),#0a4a4a)}
.tabs{display:flex;gap:8px;margin:14px 0 4px}
.tab{flex:1;width:auto;margin:0;padding:10px;background:#f1e2bd;color:var(--ink);border:1.5px solid #c9a24a;border-radius:999px}
.tab.on{background:var(--maroon);color:var(--zari)}
.msg {
    display: block !important;
    color: #1a0808 !important;
    -webkit-text-fill-color: #1a0808 !important;
    background: #fff0e8 !important;
    border-left: 5px solid #8c1230 !important;
    padding: 12px 15px !important;
    border-radius: 6px;
    margin: 8px 0;
    font-size: 16px !important;
    font-weight: 700 !important;
    opacity: 1 !important;
    visibility: visible !important;
}
.card{background:#fffdf7;border:1px solid #e3c987;padding:12px;border-radius:10px;margin:10px 0;white-space:pre-line}
.card a{color:var(--kumkum);font-weight:600}
img.res,video.res{max-width:100%;border-radius:12px;border:2px solid var(--zari)}
#camwrap{position:relative}
#cam{width:100%;border-radius:12px;border:2px solid var(--zari);background:#000;transform:scaleX(-1)}
#livebox{white-space:pre-line}
.note{font-size:14px;color:#6b4a3a}
.hide{display:none}
/* ---------- NatyaSense welcome page ---------- */
body:has(.landing) {
    background:
        radial-gradient(ellipse at 50% 15%, rgba(184,91,35,.65), transparent 48%),
        linear-gradient(115deg, #210b08, #54200f 48%, #210b08);
    color: #fff0cf;
}

body:has(.landing) header,
body:has(.landing) .zari {
    display: none;
}

body:has(.landing) main {
    max-width: 100%;
    padding: 0;
    margin: 0;
}

.landing {
    min-height: 100vh;
    position: relative;
    overflow: hidden;
    padding: 28px 5vw 45px;
    isolation: isolate;
    background:
        radial-gradient(ellipse at 50% 5%, rgba(255,184,91,.22), transparent 40%),
        linear-gradient(180deg, rgba(30,9,5,.25), rgba(25,7,4,.8));
}

.landing::before,
.landing::after {
    content: "";
    position: absolute;
    z-index: -1;
    color: rgba(230,170,76,.12);
    font-size: clamp(220px, 35vw, 460px);
    line-height: 1;
    top: 20%;
    pointer-events: none;
}

.landing::before { left: -100px; }
.landing::after { right: -100px; }

.landing-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    max-width: 1250px;
    margin: 0 auto;
}

.landing-brand {
    color: #f5c66d;
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 25px;
    text-decoration: none;
}

.landing-nav a {
    color: #f5dca8;
    text-decoration: none;
    margin-left: 18px;
}

.landing-hero {
    text-align: center;
    padding: 45px 10px 38px;
}

.landing-lotus {
    color: #e8b957;
    font-size: 55px;
    line-height: 1;
}

.landing h1 {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: clamp(48px, 7vw, 82px);
    font-weight: 600;
    letter-spacing: 2px;
    color: #f4cf83;
    margin: 8px 0 0;
}

.landing-tagline {
    color: #f2d49b;
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: clamp(18px, 2.5vw, 25px);
    letter-spacing: 2px;
    margin: 8px 0;
}

.landing-subtitle {
    color: #f8e8ce;
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 22px;
    font-style: italic;
    margin-top: 18px;
}

.practice-grid {
    width: 100%;
    max-width: 1500px;
    box-sizing: border-box;
    margin: 0 auto;
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 28px;
}

.practice-card {
    min-height: 350px;
    padding: 30px 18px;
    text-align: center;
    border: 1px solid #d8ad62;
    border-radius: 22px;
    box-shadow: 0 15px 35px rgba(0,0,0,.35), inset 0 0 20px rgba(255,220,150,.06);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.practice-card.mudra { background: linear-gradient(145deg,#67201e,#3e1016); }
.practice-card.pada { background: linear-gradient(145deg,#075044,#092c27); }
.practice-card.pose { background: linear-gradient(145deg,#492044,#28132f); }

.practice-icon {
    width: 112px !important;
    height: 112px !important;
    min-width: 112px;
    min-height: 112px;
    display: block !important;
    border: 1.5px solid #e7bd6b;
    border-radius: 50%;
    overflow: hidden !important;
    margin: 0 auto 18px;
    padding: 0;
    box-sizing: border-box;
    flex-shrink: 0;
}

.practice-icon img {
    width: 100% !important;
    height: 100% !important;
    max-width: 100% !important;
    max-height: 100% !important;
    object-fit: cover !important;
    object-position: center;
    border-radius: 50%;
    display: block !important;
}

.practice-card h2 {
    color: #fff1d5;
    font-size: 34px;
    margin: 0 0 8px;
}

.practice-card p {
    color: #f1dfc8;
    font-size: 16px;
    min-height: 50px;
    max-width: 250px;
    margin: 0 auto 20px;
}

.practice-card a {
    display: inline-block;
    width: min(210px, 100%);
    padding: 12px 18px;
    border-radius: 30px;
    background: linear-gradient(90deg,#e8bd70,#f4d99a);
    color: #3b1c0e;
    font-size: 17px;
    font-weight: 600;
    text-decoration: none;
    transition: transform .2s, box-shadow .2s;
}

.practice-card a:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 18px rgba(0,0,0,.3);
}

.landing-footer {
    text-align: center;
    color: #e8c88e;
    margin: 42px 0 0;
    letter-spacing: 2px;
}

@media(max-width:760px) {
    .practice-card { min-height: 300px; }
    .landing-hero { padding-top: 35px; }
    .landing-nav a { margin-left: 9px; }
}
@media(max-width:480px){.logo{font-size:26px}h2,h3{font-size:24px}}
@media(prefers-reduced-motion:no-preference){.logo{animation:glow 3s ease-in-out 1}@keyframes glow{from{opacity:.3}to{opacity:1}}}

/* Shared temple background on welcome, login, register and practice pages */
body{background:linear-gradient(rgba(25,10,5,.68),rgba(25,10,5,.78)),url('/static/images/natyasense-bg.jpg') center center/cover fixed no-repeat;color:#fff0cf}
body:has(.landing) header,body:has(.landing) .zari{display:none}
body:has(.landing) main{max-width:100%;padding:0;margin:0}
body:has(.landing) .landing{background:transparent}
.auth-wrap{min-height:calc(100vh - 32px);display:grid;place-items:center;padding:35px 16px}
.auth-panel{width:min(460px,100%);background:rgba(255,246,227,.96);border:2px solid #d4a017;border-radius:22px;padding:28px;box-shadow:0 15px 45px #0009}
.auth-panel h2{ text-align:center }
.auth-panel .btn-row{display:flex;gap:12px;flex-wrap:wrap}
.auth-panel .btn-row a{flex:1}
.selection-wrap {
    min-height: calc(100vh - 32px);
    width: 100%;
    box-sizing: border-box;
    padding: 45px 2vw;
}
.selection-title{text-align:center;color:#f4cf83;font:600 clamp(34px,5vw,54px) 'Cormorant Garamond',Georgia,serif}
.selection-sub{text-align:center;color:#f8e8ce;margin:0 0 34px}
.practice-card{min-height:330px;padding:28px 20px;text-align:center;border:1px solid #d8ad62;border-radius:22px;box-shadow:0 15px 35px #0006;display:flex;flex-direction:column;align-items:center;justify-content:center}
.practice-card.mudra{background:linear-gradient(145deg,#67201e,#3e1016)}.practice-card.pada{background:linear-gradient(145deg,#075044,#092c27)}.practice-card.pose{background:linear-gradient(145deg,#492044,#28132f)}
.practice-icon{width:112px;height:112px;display:grid;place-items:center;border:1.5px solid #e7bd6b;border-radius:50%;color:#f2cc7d;margin-bottom:18px;background:rgba(0,0,0,.12)}
.practice-icon svg{width:70px;height:70px;stroke:currentColor;fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.practice-card h2{color:#fff1d5;font-size:32px;margin:0 0 8px}.practice-card p{color:#f1dfc8;min-height:50px;max-width:250px;margin:0 auto 18px}
.practice-card a{display:inline-block;width:min(210px,100%);padding:11px;border-radius:30px;background:linear-gradient(90deg,#e8bd70,#f4d99a);color:#3b1c0e;font-weight:600;text-decoration:none}
.practice-panel{background:rgba(255,246,227,.97);border:2px solid #d4a017;border-radius:20px;padding:22px;margin:20px auto;max-width:760px;box-shadow:0 12px 35px #0008}
.practice-panel{color:#2b1a14}
.practice-panel h2,.practice-panel h3,.practice-panel h4,
.practice-panel p,.practice-panel label,.practice-panel .card,
.practice-panel .note{color:#2b1a14}
.practice-panel .card{background:#fffdf7;color:#2b1a14}
.result-panel{text-align:center}
.result-name{font-family:'Cormorant Garamond',Georgia,serif;font-size:36px;font-weight:700;color:#6b1b27;margin:18px 0}
.result-feedback{text-align:left;font-size:18px;line-height:1.6}
.result-feedback p{margin:8px 0 0;color:#2b1a14}

.practice-panel .back-link{color:#6b1b27;font-weight:700}
.practice-panel input,.practice-panel select{color:#2b1a14}

.back-link{color:#f5dca8;text-decoration:none;border-bottom:1px solid #d4a017}
@media(max-width:760px){.practice-grid{grid-template-columns:1fr;max-width:420px}.practice-card{min-height:285px}}

/* NatyaSense shared temple background */
html, body {
    min-height: 100%;
    background-color: #24100b;
}

body {
    background-image:
        linear-gradient(rgba(18,9,8,.58), rgba(18,9,8,.68)),
        url('/static/images/natyasense-bg.jpg') !important;
    background-position: center center !important;
    background-size: cover !important;
    background-repeat: no-repeat !important;
    background-attachment: fixed !important;
}
body:has(.landing) header, body:has(.landing) .zari { display:none; }
body:has(.landing) main { max-width:100%; padding:0; margin:0; }
body:has(.landing) .landing { background:transparent; }


/* Password visibility controls and camera preview */
.password-field{display:flex;align-items:stretch;gap:8px}
.password-field input{flex:1;min-width:0}
.password-field .show-password{width:auto;min-width:125px;margin:5px 0;padding:8px 12px;font-size:14px;background:#4a0816}
#cam{display:block;width:100%;min-height:220px;object-fit:cover;background:#090909;transform:scaleX(-1)}
#camwrap{overflow:hidden;border-radius:12px;background:#090909}
#capture{background:linear-gradient(#0f6b6b,#084848)}
/* Correct shared temple background for all pages */
html {
    min-height: 100%;
    background-color: #24100b;
}

body {
    min-height: 100vh;
    margin: 0;
    background-color: #24100b;
    background-image:
        linear-gradient(rgba(18,9,8,.40), rgba(18,9,8,.50)),
        url('/static/images/natyasense-bg.jpg');
    background-position: center center;
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
/* Golden Bharatanatyam illustrations */
.practice-icon {
  color:#f5cf79 !important;
  background:rgba(20,10,8,.20);
  border:2px solid #e8bd70;
  box-shadow:0 0 0 5px rgba(232,189,112,.10), inset 0 0 18px rgba(245,207,121,.12);
}
.practice-icon svg { width:76px; height:76px; display:block; overflow:visible; }
.practice-card h2, .practice-card p { text-shadow:0 1px 5px rgba(0,0,0,.35); }
.selection-wrap { padding-top:26px; padding-bottom:38px; }
.selection-title, .selection-sub { text-shadow:0 2px 8px rgba(0,0,0,.7); }
.practice-card { backdrop-filter:blur(3px); }

label {
    display: block;
    font-weight: 700;
    margin-top: 10px;
    color: #1a0808 !important;
    opacity: 1 !important;
    font-size: 17px !important;
}
.msg {
    color: #4a1025 !important;
    background-color: #ffe4ec !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    opacity: 1 !important;
    text-shadow: none !important;
    padding: 15px 20px !important;
    border-left: 5px solid #8b1e3f !important;
    border-radius: 10px !important;
}
/* Full-screen layout for every page */
* {
    box-sizing: border-box;
}

html, body {
    width: 100%;
    min-height: 100%;
    margin: 0;
    padding: 0;
}

body {
    min-height: 100vh;
}

main {
    width: 100%;
    max-width: 760px;
    min-height: calc(100vh - 32px);
    margin: 0 auto;
    padding: 0 14px 40px;
}
</style></head><body>
<div class="zari"></div>
<header><a class="logo" href="/">NatyaSense<small>your practice companion for Bharatanatyam</small></a>
<nav>{% if session.uid %}<a href="/practice">Practice</a><a href="/logout">Logout</a>
{% else %}<a href="/login">Login</a><a href="/register">Register</a>{% endif %}</nav></header>
<main>
{% for m in get_flashed_messages() %}<p class="msg">{{ m }}</p>{% endfor %}
{{ body|safe }}
</main><div class="zari"></div></body></html>"""

def page(tpl, **ctx):
    return render_template_string(BASE, body=render_template_string(tpl, **ctx))

# ---------------------------------------------------------------- welcome page

HOME_TPL = """
<div class="landing">
  <section class="landing-hero">
    <div class="landing-lotus">❀</div>
    <h1>NatyaSense</h1>
    <p class="landing-tagline">Preserve &nbsp; ✦ &nbsp; Practice &nbsp; ✦ &nbsp; Perfect</p>
    <p class="landing-subtitle">Your AI companion for Bharatanatyam</p>
    <div style="width:min(420px,90%);margin:30px auto;display:flex;gap:14px;flex-wrap:wrap">
      <a class="btn" href="{{ url_for('login') }}">Login</a>
      <a class="btn" href="{{ url_for('register') }}">Register</a>
    </div>
  </section>
</div>
"""

PRACTICE_HOME_TPL = """
<div class="selection-wrap">
  <h1 class="selection-title">Namaste, {{ session.name }}</h1>
  <p class="selection-sub">Choose what you would like to practise today</p>
  <section class="practice-grid">
    <div class="practice-card mudra">
      <div class="practice-icon"><img src="{{ url_for('static', filename='images/mudra-icon.jpg') }}" alt="Mudra"></div>
      <h2>Mudra</h2><p>Check and correct your Bharatanatyam hand gestures.</p>
      <a href="{{ url_for('upload', task='mudra') }}">Start Practice &nbsp; →</a>
    </div>
    <div class="practice-card pada">
      <div class="practice-icon"><img src="{{ url_for('static', filename='images/pada-icon.jpg') }}" alt="Pada Bheda"></div>
      <h2>Pada Bheda</h2><p>Check and correct your foot positions and movements.</p>
      <a href="{{ url_for('upload', task='padabheda') }}">Start Practice &nbsp; →</a>
    </div>
    <div class="practice-card pose">
      <div class="practice-icon"><img src="{{ url_for('static', filename='images/pose-icon.jpg') }}" alt="Pose"></div>
      <h2>Pose</h2><p>Check and correct your Bharatanatyam body posture.</p>
      <a href="{{ url_for('upload', task='pose') }}">Start Practice &nbsp; →</a>
    </div>
  </section>
</div>
"""

@app.route("/")
def home():
    if "uid" in session:
        return redirect(url_for("practice_home"))
    return page(HOME_TPL)

@app.route("/practice")
@login_required
def practice_home():
    return page(PRACTICE_HOME_TPL)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"].strip()
        email = request.form["email"].strip().lower()
        pw = request.form["password"]
        if not name or not email or len(pw) < 6:
            flash("Fill all fields. The password needs at least 6 characters.")
        else:
            try:
                db().execute("INSERT INTO users(name,email,password) VALUES(?,?,?)",
                             (name, email, generate_password_hash(pw)))
                db().commit()
                flash("Registered. Please log in.")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("This email is already registered. Log in instead.")
    return page("""<div class="panel"><h2>Join the class</h2><form method="post">
    <label>Name</label><input name="name" autocomplete="name" required>
    <label>Email</label><input name="email" type="email" autocomplete="email" required>
    <label>Password (6 or more characters)</label>
    <div class="password-field"><input id="register_password" name="password" type="password" autocomplete="new-password" required>
    <button type="button" class="show-password" data-password="register_password">Show password</button></div>
    <button>Create account</button></form></div>
    <script>
    document.querySelectorAll('.show-password').forEach(btn=>btn.addEventListener('click',()=>{
      const field=document.getElementById(btn.dataset.password);
      const visible=field.type==='password'; field.type=visible?'text':'password';
      btn.textContent=visible?'Hide password':'Show password';
    }));
    </script>""")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = db().execute("SELECT * FROM users WHERE email=?",
                         (request.form["email"].strip().lower(),)).fetchone()
        if u and check_password_hash(u["password"], request.form["password"]):
            session["uid"], session["name"] = u["id"], u["name"]
            return redirect(url_for("practice_home"))
        flash("Wrong email or password.")
    return page("""<div class="panel"><h2>Welcome back</h2><form method="post">
    <label>Email</label><input name="email" type="email" autocomplete="email" required>
    <label>Password</label>
    <div class="password-field"><input id="login_password" name="password" type="password" autocomplete="current-password" required>
    <button type="button" class="show-password" data-password="login_password">Show password</button></div>
    <button>Log in</button></form>
    <p style="text-align:center;margin-top:12px"><a href="{{ url_for('forgot_password') }}">Forgot Password?</a></p>
    </div>
    <script>
    document.querySelectorAll('.show-password').forEach(btn=>btn.addEventListener('click',()=>{
      const field=document.getElementById(btn.dataset.password);
      const visible=field.type==='password'; field.type=visible?'text':'password';
      btn.textContent=visible?'Hide password':'Show password';
    }));
    </script>""")

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        user = db().execute("SELECT id,email FROM users WHERE email=?", (email,)).fetchone()
        # Do not disclose whether an email address is registered.
        if user:
            serializer = URLSafeTimedSerializer(app.secret_key)
            token = serializer.dumps(email, salt="natyasense-password-reset")
            reset_url = url_for("reset_password", token=token, _external=True)
            try:
                send_password_reset_email(email, reset_url)
            except Exception:
                app.logger.exception("Could not send password reset email")
                flash("Password-reset email could not be sent. Check the email settings in the setup instructions.")
                return page("""<div class="panel"><h2>Forgot Password</h2>
                <form method="post"><label>Registered email</label><input name="email" type="email" required>
                <button>Send reset link</button></form></div>""")
        flash("If that email is registered, a password-reset link has been sent. Check your inbox.")
        return redirect(url_for("login"))
    return page("""<div class="panel"><h2>Forgot Password</h2>
    <p>Enter the email address you used to register. We will send a secure reset link.</p>
    <form method="post"><label>Registered email</label>
    <input name="email" type="email" autocomplete="email" required>
    <button>Send reset link</button></form>
    <p><a href="{{ url_for('login') }}">Back to login</a></p></div>""")

@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    serializer = URLSafeTimedSerializer(app.secret_key)
    try:
        email = serializer.loads(token, salt="natyasense-password-reset",
                                 max_age=RESET_TOKEN_MAX_AGE)
    except SignatureExpired:
        flash("This reset link has expired. Please request a new one.")
        return redirect(url_for("forgot_password"))
    except BadSignature:
        flash("This reset link is invalid. Please request a new one.")
        return redirect(url_for("forgot_password"))

    user = db().execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
    if not user:
        flash("This account could not be found. Please request a new reset link.")
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        if len(password) < 6:
            flash("The new password must contain at least 6 characters.")
        elif password != confirm:
            flash("The passwords do not match.")
        else:
            db().execute("UPDATE users SET password=? WHERE id=?",
                         (generate_password_hash(password), user["id"]))
            db().commit()
            flash("Your password has been changed. Please log in.")
            return redirect(url_for("login"))

    return page("""<div class="panel"><h2>Set a new password</h2>
    <form method="post">
    <label>New password (at least 6 characters)</label>
    <div class="password-field"><input id="new_password" name="password" type="password" autocomplete="new-password" required>
    <button type="button" class="show-password" data-password="new_password">Show password</button></div>
    <label>Confirm new password</label>
    <div class="password-field"><input id="confirm_password" name="confirm_password" type="password" autocomplete="new-password" required>
    <button type="button" class="show-password" data-password="confirm_password">Show password</button></div>
    <button>Change password</button></form></div>
    <script>
    document.querySelectorAll('.show-password').forEach(btn=>btn.addEventListener('click',()=>{
      const field=document.getElementById(btn.dataset.password);
      const visible=field.type==='password'; field.type=visible?'text':'password';
      btn.textContent=visible?'Hide password':'Show password';
    }));
    </script>""")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# ---------------------------------------------------------------- practice
@app.route("/upload")
@login_required
def upload_legacy():
    return redirect(url_for("practice_home"))

@app.route("/practice/<task>", methods=["GET", "POST"])
@login_required
def upload(task):
    # The category comes only from the URL route, never from a user-editable form field.
    if task not in TITLES:
        return redirect(url_for("practice_home"))
    if request.method == "POST":
        f = request.files.get("photo")
        ext = os.path.splitext(f.filename)[1].lower() if f else ""
        target = (
    canonical_pada(request.form.get("target")) or None
    ) if task == "padabheda" else None
        if target and target not in TIPS:
            target = None
        if not f or ext not in IMG_EXT | VID_EXT:
            flash("Choose a JPG or PNG photo, or an MP4, MOV, AVI or WEBM video.")
        elif ext in IMG_EXT:
            name = uuid.uuid4().hex + ext
            path = os.path.join(UPLOAD_DIR, name)
            f.save(path)
            d = evaluate(task, path, target)
            return save_result(task, name, target, d, "")
        else:
            vpath = os.path.join(UPLOAD_DIR, "tmp_" + uuid.uuid4().hex + ext)
            f.save(vpath)
            try:
                d, thumb, err = run_video(task, vpath, target)
            finally:
                if os.path.exists(vpath):
                    os.remove(vpath)
            if err:
                flash(err)
            else:
                return save_result(task, thumb, target, d, " (video, up to 30 s)")
    rows = db().execute("SELECT * FROM results WHERE user_id=? AND task=? ORDER BY id DESC LIMIT 5",
                        (session["uid"], task)).fetchall()
    return page(UPLOAD_TPL, task=task, title=TITLES[task], classes=list(PADA_DISPLAY.values()), rows=rows)

def save_result(task, filename, target, d, note):
    print("FEEDBACK DEBUG:", task, "LABEL:", d.get("label"), "SCORE:", d.get("score"), "CONF:", d.get("conf"), "MSGS:", d.get("msgs"))
    cur = db().execute(
        "INSERT INTO results(user_id,filename,target,predicted,confidence,feedback,task) "
        "VALUES(?,?,?,?,?,?,?)",
        (session["uid"], filename, target, d["label"], d["conf"], to_text(task, d, note), task))
    db().commit()
    return redirect(url_for("result", rid=cur.lastrowid))

@app.route("/live_frame/<task>", methods=["POST"])
@login_required
def live_frame(task):
    f = request.files.get("frame")
    if not f or task not in TITLES:
        return jsonify(error="bad request"), 400
    target = (
    canonical_pada(request.form.get("target")) or None
    ) if task == "padabheda" else None
    fd, tmp = tempfile.mkstemp(suffix=".jpg", dir=UPLOAD_DIR)
    os.close(fd)
    try:
        f.save(tmp)
        d = evaluate(task, tmp, target)
    finally:
        os.remove(tmp)
    return jsonify(label=d["label"], feedback=feedback_for_result(task, d))

@app.route("/result/<int:rid>")
@login_required
def result(rid):
    r = db().execute("SELECT * FROM results WHERE id=? AND user_id=?",
                     (rid, session["uid"])).fetchone()
    if not r:
        return redirect(url_for("practice_home"))
    task = r["task"] if "task" in r.keys() and r["task"] in TITLES else "mudra"
    return page("""<div class="practice-panel result-panel">
    <h3>Your Practice Result</h3>
    <div class="result-name">{{ r.predicted }}</div>
    <div class="card result-feedback"><b>Correction Feedback</b><p>{{ r.feedback }}</p></div>
    <a class="btn" href="{{ url_for('upload', task=task) }}">Practice again</a>
    <a class="back-link" href="/practice">Back to categories</a>
    </div>""", r=r, task=task)

@app.route("/uploads/<fn>")
@login_required
def uploads(fn):
    return send_from_directory(UPLOAD_DIR, fn)

UPLOAD_TPL = """
<div class="practice-panel">
<a class="back-link" href="/practice">← Back to practice categories</a>
<h2>{{ title }} Practice</h2>
<p>Namaste, {{ session.name }}. Upload a photo, a video up to 30 seconds, or use your live camera.</p>
{% if task == 'padabheda' %}
<label for="target">Which Pada Bheda are you practising? (optional)</label>
<select id="target" name="target"><option value="">Not sure</option>{% for c in classes %}<option value="{{ c }}">{{ c }}</option>{% endfor %}</select>
{% else %}<input type="hidden" id="target" value="">{% endif %}
<div class="tabs" role="tablist">
<button type="button" class="tab on" id="t1" role="tab">Photo or video</button>
<button type="button" class="tab" id="t2" role="tab">Live camera</button></div>
<section id="p1"><form method="post" enctype="multipart/form-data" id="upform">
<input type="hidden" name="target" id="target_post">
<label for="photo">Choose a photo, or a video of up to 30 seconds</label>
<input type="file" id="photo" name="photo" accept="image/jpeg,image/png,video/mp4,video/quicktime,video/x-msvideo,video/webm,video/x-matroska" required>
<p class="note">Keep the {{ title|lower }} clearly visible. Videos longer than 30 seconds are not accepted.</p>
<button id="go">Check my practice</button></form></section>
<section id="p2" class="hide">
<div id="camwrap"><video id="cam" playsinline muted></video></div><canvas id="snap" class="hide"></canvas>
<button type="button" id="start">Start camera</button>
<button type="button" id="capture" class="alt hide">Capture and check</button>
<button type="button" id="stop" class="alt hide">Stop camera</button>
<div class="card" id="livebox">Press Start camera and allow camera permission. Live feedback updates every couple of seconds.</div>
<p class="note">Camera access requires permission and a secure context. For local testing, use http://127.0.0.1:5000 or http://localhost:5000.</p>
</section></div>
<div class="practice-panel"><h4>Recent {{ title }} practice</h4>
{% for r in rows %}<div class="card"><a href="/result/{{ r.id }}">{{ r.predicted }}</a> on {{ r.created }}</div>
{% else %}<p>No {{ title|lower }} practice yet. Upload your first photo or video.</p>{% endfor %}</div>
<script>
const $=id=>document.getElementById(id);
const TASK={{ task|tojson }};
let stream=null,timer=null,busy=false;
function tab(n){
  $('p1').classList.toggle('hide',n!==1);$('p2').classList.toggle('hide',n!==2);
  $('t1').classList.toggle('on',n===1);$('t2').classList.toggle('on',n===2);
  if(n===1)stopCam();
}
$('t1').addEventListener('click',()=>tab(1));
$('t2').addEventListener('click',()=>tab(2));
$('upform').addEventListener('submit',()=>{
  $('target_post').value=$('target').value;
  $('go').textContent='Checking, please wait...';$('go').disabled=true;
});
async function startCam(){
  const box=$('livebox');
  if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){
    box.textContent='This browser does not allow camera access here. Open the app using http://127.0.0.1:5000 or http://localhost:5000, then try again.';
    return;
  }
  $('start').disabled=true;box.textContent='Requesting camera permission...';
  try{
    stream=await navigator.mediaDevices.getUserMedia({
      audio:false,video:{facingMode:{ideal:'user'},width:{ideal:640},height:{ideal:480}}
    });
    const video=$('cam');video.srcObject=stream;
    await video.play();
    $('start').classList.add('hide');$('capture').classList.remove('hide');
    $('stop').classList.remove('hide');box.textContent='Camera is on. Position yourself clearly in the frame.';
    timer=setInterval(tick,1800);
  }catch(e){
    if(stream){stream.getTracks().forEach(track=>track.stop());stream=null;}
    box.textContent='Could not start camera: '+(e.name||'Error')+' — '+(e.message||'Check browser permission and close other apps using the camera.');
  }finally{$('start').disabled=false;}
}
function stopCam(){
  if(timer){clearInterval(timer);timer=null;}
  if(stream){stream.getTracks().forEach(track=>track.stop());stream=null;}
  const video=$('cam');if(video)video.srcObject=null;
  $('capture').classList.add('hide');$('stop').classList.add('hide');
  $('start').classList.remove('hide');$('start').disabled=false;
}
function frameBlob(){
  const video=$('cam'),canvas=$('snap');
  if(!video.videoWidth)return Promise.resolve(null);
  canvas.width=video.videoWidth;canvas.height=video.videoHeight;
  canvas.getContext('2d').drawImage(video,0,0,canvas.width,canvas.height);
  return new Promise(resolve=>canvas.toBlob(resolve,'image/jpeg',0.9));
}
async function tick(){
  if(busy)return;busy=true;
  try{
    const blob=await frameBlob();if(!blob)return;
    const form=new FormData();form.append('frame',blob,'camera-frame.jpg');
    form.append('target',$('target').value);
    const response=await fetch('/live_frame/'+TASK,{method:'POST',body:form});
    const data=await response.json();if(!response.ok)throw new Error(data.error||'Camera request failed');
    show(data);
  }catch(e){$('livebox').textContent='Live recognition error: '+e.message;}
  finally{busy=false;}
}
async function captureAndCheck(){
  if(busy)return;
  busy=true;$('capture').disabled=true;$('livebox').textContent='Capturing and checking your practice...';
  try{
    const blob=await frameBlob();if(!blob)throw new Error('Camera image is not ready. Wait a moment and try again.');
    const form=new FormData();form.append('photo',blob,'camera-capture.jpg');
    form.append('target',$('target').value);
    const response=await fetch('/practice/'+TASK,{method:'POST',body:form,redirect:'follow'});
    if(!response.ok)throw new Error('Could not submit the captured image. Please try again.');
    window.location.href=response.url;
  }catch(e){$('livebox').textContent=e.message;}
  finally{busy=false;$('capture').disabled=false;}
}
function show(data){
  if(data.error){$('livebox').textContent=data.error;return;}
  $('livebox').textContent=(data.label&&data.label!=='none'?data.label+'\\n\\n':'')+
    (data.feedback||'Please try again with the movement clearly visible.');
}
$('start').addEventListener('click',startCam);
$('stop').addEventListener('click',stopCam);
$('capture').addEventListener('click',captureAndCheck);
window.addEventListener('beforeunload',stopCam);
</script>
"""

if __name__ == "__main__":
    app.run(debug=False, threaded=True)