import os, uuid, sqlite3, tempfile, threading
from collections import Counter
from functools import wraps

import cv2
from flask import (Flask, request, redirect, url_for, session, g, flash,
                   render_template_string, send_from_directory, jsonify)
from werkzeug.security import generate_password_hash, check_password_hash

from src.predict_pada import predict_pada
from src.analyze import analyze

app = Flask(__name__)
app.secret_key = "change-this-later"
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024
UPLOAD_DIR, DB = "uploads", "natyasense.db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
LOCK = threading.Lock()          # models run one request at a time
IMG_EXT = {".jpg", ".jpeg", ".png"}
VID_EXT = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
MAX_SECONDS, MAX_FRAMES = 30, 20

# Draft tips: check these with your guru and edit the text.
TIPS = {
    "Anchitha": "Keep the heels on the floor and lift the toes upward.",
    "Hayatham": "Add your guru's description for Hayatham here.",
    "Kunchita": "Press the toes on the floor and raise the heels.",
    "Samapada": "Keep both feet flat on the floor, together, with weight even.",
    "Ekapada": "Keep your weight on one foot and lift the other foot.",
}
TITLES = {"mudra": "Mudra", "pose": "Pose", "padabheda": "Pada Bheda"}

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

def login_required(f):
    @wraps(f)
    def w(*a, **k):
        if "uid" not in session:
            return redirect(url_for("login"))
        return f(*a, **k)
    return w

# ---------------------------------------------------------------- analysis
def pada_feedback(target, pred, conf):
    if conf < 0.5:
        return ("Not sure about this one. Retake it with both feet clearly "
                "visible, good light, feet in the lower part of the frame.")
    if target and target == pred:
        return "Good! Your feet match " + target + ". " + TIPS[pred]
    if target:
        return ("You are practicing " + target + ", but it looks like " + pred +
                ". To correct it: " + TIPS[target])
    return "Detected " + pred + ". " + TIPS[pred]

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

def to_text(task, d, note=""):
    t = TITLES[task] + note
    if task == "padabheda":
        return t + "\n" + d["msgs"][0]
    if d["label"] == "none" or d["score"] is None:
        return t + "\n" + "\n".join(d["msgs"])
    t += "\nScore: " + str(d["score"]) + " (" + str(d["tier"]) + ")"
    if d["msgs"]:
        t += "\nTo improve:\n- " + "\n- ".join(d["msgs"])
    else:
        t += "\nVery good, no corrections needed."
    return t

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
label{display:block;font-weight:600;margin-top:10px}
input,select{width:100%;padding:11px;margin:5px 0;border:1.5px solid #c9a24a;border-radius:8px;background:#fffdf7;font:inherit}
input:focus,select:focus,button:focus-visible,.tab:focus-visible{outline:3px solid var(--peacock);outline-offset:2px}
button,.btn{display:inline-block;width:100%;padding:12px;margin:8px 0;background:linear-gradient(var(--kumkum),#8c0d22);
 color:#fff;border:1.5px solid var(--zari);border-radius:999px;font:600 17px Mukta,Arial;cursor:pointer;text-align:center;text-decoration:none}
button.alt{background:linear-gradient(var(--peacock),#0a4a4a)}
.tabs{display:flex;gap:8px;margin:14px 0 4px}
.tab{flex:1;width:auto;margin:0;padding:10px;background:#f1e2bd;color:var(--ink);border:1.5px solid #c9a24a;border-radius:999px}
.tab.on{background:var(--maroon);color:var(--zari)}
.msg{background:#fde2e2;border-left:5px solid var(--kumkum);padding:9px 12px;border-radius:6px;margin:8px 0}
.card{background:#fffdf7;border:1px solid #e3c987;padding:12px;border-radius:10px;margin:10px 0;white-space:pre-line}
.card a{color:var(--kumkum);font-weight:600}
img.res,video.res{max-width:100%;border-radius:12px;border:2px solid var(--zari)}
#camwrap{position:relative}
#cam{width:100%;border-radius:12px;border:2px solid var(--zari);background:#000;transform:scaleX(-1)}
#livebox{white-space:pre-line}
.note{font-size:14px;color:#6b4a3a}
.hide{display:none}
@media(max-width:480px){.logo{font-size:26px}h2,h3{font-size:24px}}
@media(prefers-reduced-motion:no-preference){.logo{animation:glow 3s ease-in-out 1}@keyframes glow{from{opacity:.3}to{opacity:1}}}
</style></head><body>
<div class="zari"></div>
<header><a class="logo" href="/">NatyaSense<small>your practice companion for Bharatanatyam</small></a>
<nav>{% if session.uid %}<a href="/upload">Practice</a><a href="/logout">Logout</a>
{% else %}<a href="/login">Login</a><a href="/register">Register</a>{% endif %}</nav></header>
<main>
{% for m in get_flashed_messages() %}<p class="msg">{{ m }}</p>{% endfor %}
{{ body|safe }}
</main><div class="zari"></div></body></html>"""

def page(tpl, **ctx):
    return render_template_string(BASE, body=render_template_string(tpl, **ctx))

# ---------------------------------------------------------------- accounts
@app.route("/")
def home():
    return redirect(url_for("upload" if "uid" in session else "login"))

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
    <label>Name</label><input name="name" autocomplete="name">
    <label>Email</label><input name="email" type="email" autocomplete="email">
    <label>Password (6 or more characters)</label><input name="password" type="password" autocomplete="new-password">
    <button>Create account</button></form></div>""")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = db().execute("SELECT * FROM users WHERE email=?",
                         (request.form["email"].strip().lower(),)).fetchone()
        if u and check_password_hash(u["password"], request.form["password"]):
            session["uid"], session["name"] = u["id"], u["name"]
            return redirect(url_for("upload"))
        flash("Wrong email or password.")
    return page("""<div class="panel"><h2>Welcome back</h2><form method="post">
    <label>Email</label><input name="email" type="email" autocomplete="email">
    <label>Password</label><input name="password" type="password" autocomplete="current-password">
    <button>Log in</button></form></div>""")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------------------------------------------------------------- practice
@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        f = request.files.get("photo")
        ext = os.path.splitext(f.filename)[1].lower() if f else ""
        task = request.form.get("task", "mudra")
        target = (request.form.get("target") or None) if task == "padabheda" else None
        if task not in TITLES:
            task = "mudra"
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
                os.remove(vpath)
            if err:
                flash(err)
            else:
                return save_result(task, thumb, target, d, " (video, up to 30 s)")
    rows = db().execute("SELECT * FROM results WHERE user_id=? ORDER BY id DESC LIMIT 5",
                        (session["uid"],)).fetchall()
    return page(UPLOAD_TPL, classes=list(TIPS), rows=rows)

def save_result(task, filename, target, d, note):
    cur = db().execute(
        "INSERT INTO results(user_id,filename,target,predicted,confidence,feedback) "
        "VALUES(?,?,?,?,?,?)",
        (session["uid"], filename, target, d["label"], d["conf"], to_text(task, d, note)))
    db().commit()
    return redirect(url_for("result", rid=cur.lastrowid))

@app.route("/live_frame", methods=["POST"])
@login_required
def live_frame():
    f = request.files.get("frame")
    task = request.form.get("task", "mudra")
    if not f or task not in TITLES:
        return jsonify(error="bad request"), 400
    target = (request.form.get("target") or None) if task == "padabheda" else None
    fd, tmp = tempfile.mkstemp(suffix=".jpg", dir=UPLOAD_DIR)
    os.close(fd)
    try:
        f.save(tmp)
        d = evaluate(task, tmp, target)
    finally:
        os.remove(tmp)
    return jsonify(label=d["label"], conf=round(d["conf"] * 100), score=d["score"],
                   tier=d["tier"], msgs=d["msgs"])

@app.route("/result/<int:rid>")
@login_required
def result(rid):
    r = db().execute("SELECT * FROM results WHERE id=? AND user_id=?",
                     (rid, session["uid"])).fetchone()
    if not r:
        return redirect(url_for("upload"))
    return page("""<div class="panel"><h3>Your result</h3>
    <img class="res" src="/uploads/{{ r.filename }}" alt="your photo">
    <div class="card"><b>Detected:</b> {{ r.predicted }}<br>
    <b>Confidence:</b> {{ (r.confidence*100)|round|int }}%</div>
    <div class="card"><b>Feedback from your practice companion</b><br>{{ r.feedback }}</div>
    <a class="btn" href="/upload">Practice again</a></div>""", r=r)

@app.route("/uploads/<fn>")
@login_required
def uploads(fn):
    return send_from_directory(UPLOAD_DIR, fn)

UPLOAD_TPL = """<div class="panel"><h2>Namaste, {{ session.name }}</h2>
<label for="task">What are you practising?</label>
<select id="task"><option value="mudra">Mudra (hand gesture)</option>
<option value="pose">Pose (whole body)</option>
<option value="padabheda">Pada Bheda (feet)</option></select>
<div id="tbox" class="hide"><label for="target">Which Pada Bheda? (optional)</label>
<select id="target"><option value="">Not sure</option>
{% for c in classes %}<option>{{ c }}</option>{% endfor %}</select></div>

<div class="tabs" role="tablist">
<button type="button" class="tab on" id="t1" role="tab">Photo or video</button>
<button type="button" class="tab" id="t2" role="tab">Live camera</button></div>

<section id="p1"><form method="post" enctype="multipart/form-data" id="upform">
<input type="hidden" name="task" id="f_task"><input type="hidden" name="target" id="f_target">
<label for="photo">Choose a photo, or a video of up to 30 seconds</label>
<input type="file" id="photo" name="photo" accept="image/*,video/*" required>
<p class="note">Videos take 10 to 30 seconds to check. Keep the hand, body or feet in view.</p>
<button id="go">Check my practice</button></form></section>

<section id="p2" class="hide">
<div id="camwrap"><video id="cam" playsinline muted></video></div>
<canvas id="snap" class="hide"></canvas>
<button type="button" id="start">Start camera</button>
<button type="button" id="stop" class="alt hide">Stop camera</button>
<div class="card" id="livebox">Press Start camera. Feedback updates every couple of seconds.</div>
<p class="note">Live camera works at http://127.0.0.1:5000 on this laptop. Allow camera access when the browser asks.</p>
</section></div>

<div class="panel"><h4>Recent practice</h4>
{% for r in rows %}<div class="card"><a href="/result/{{ r.id }}">{{ r.predicted }}
({{ (r.confidence*100)|round|int }}%)</a> on {{ r.created }}</div>
{% else %}<p>Nothing yet. Upload your first photo or video.</p>{% endfor %}</div>

<script>
const $=id=>document.getElementById(id);
let stream=null,timer=null,busy=false;
function syncTask(){$('tbox').classList.toggle('hide',$('task').value!=='padabheda');}
$('task').onchange=syncTask;syncTask();
function tab(n){
  $('p1').classList.toggle('hide',n!==1);$('p2').classList.toggle('hide',n!==2);
  $('t1').classList.toggle('on',n===1);$('t2').classList.toggle('on',n===2);
  if(n===1)stopCam();
}
$('t1').onclick=()=>tab(1);$('t2').onclick=()=>tab(2);
$('upform').onsubmit=()=>{
  $('f_task').value=$('task').value;$('f_target').value=$('target').value;
  $('go').textContent='Checking, please wait...';$('go').disabled=true;
};
async function startCam(){
  try{stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user',width:640}});}
  catch(e){$('livebox').textContent='Camera not available: '+e.message;return;}
  const v=$('cam');v.srcObject=stream;await v.play();
  $('start').classList.add('hide');$('stop').classList.remove('hide');
  $('livebox').textContent='Looking...';
  timer=setInterval(tick,1500);
}
function stopCam(){
  if(timer){clearInterval(timer);timer=null;}
  if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}
  $('stop').classList.add('hide');$('start').classList.remove('hide');
}
function tick(){
  const v=$('cam'),c=$('snap');
  if(busy||!v.videoWidth)return;
  busy=true;c.width=v.videoWidth;c.height=v.videoHeight;
  c.getContext('2d').drawImage(v,0,0);
  c.toBlob(async b=>{
    const fd=new FormData();
    fd.append('frame',b,'f.jpg');fd.append('task',$('task').value);fd.append('target',$('target').value);
    try{
      const r=await fetch('/live_frame',{method:'POST',body:fd});
      show(await r.json());
    }catch(e){$('livebox').textContent='Connection lost. Press Stop camera and try again.';}
    busy=false;
  },'image/jpeg',0.85);
}
function show(j){
  let t=j.label==='none'?'':'Detected: '+j.label+' ('+j.conf+'%)\\n';
  if(j.score!==null&&j.score!==undefined)t+='Score: '+j.score+' ('+j.tier+')\\n';
  if(j.msgs&&j.msgs.length)t+=j.msgs.join('\\n');
  $('livebox').textContent=t;
}
$('start').onclick=startCam;$('stop').onclick=stopCam;
</script>"""

if __name__ == "__main__":
    app.run(debug=False, threaded=True)
