import os, uuid, sqlite3
from functools import wraps
from flask import (Flask, request, redirect, url_for, session, g,
                   flash, render_template_string, send_from_directory)
from werkzeug.security import generate_password_hash, check_password_hash
from src.predict_pada import predict_pada

app = Flask(__name__)
app.secret_key = "change-this-later"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
UPLOAD_DIR, DB = "uploads", "natyasense.db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Draft tips: please check these with your guru and edit the text.
TIPS = {
    "Anchitha": "Keep the heels on the floor and lift the toes upward.",
    "Hayatham": "Add your guru's description for Hayatham here.",
    "Kunchita": "Press the toes on the floor and raise the heels.",
    "Samapada": "Keep both feet flat on the floor, together, with weight even.",
    "Ekapada": "Keep your weight on one foot and lift the other foot.",
}

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

BASE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NatyaSense</title><style>
body{font-family:Arial;max-width:480px;margin:auto;padding:16px;background:#fdf8f0}
nav{display:flex;justify-content:space-between;margin-bottom:12px}
input,select,button{width:100%;padding:10px;margin:6px 0;box-sizing:border-box}
button{background:#b8860b;color:#fff;border:0;border-radius:6px;font-size:16px}
.msg{background:#fde2e2;padding:8px;border-radius:6px}
.card{background:#fff;padding:12px;border-radius:8px;margin:10px 0}
img{max-width:100%;border-radius:8px}
</style></head><body>
<nav><b>NatyaSense</b><span>
{% if session.uid %}<a href="/upload">Upload</a> | <a href="/logout">Logout</a>
{% else %}<a href="/login">Login</a> | <a href="/register">Register</a>{% endif %}
</span></nav>
{% for m in get_flashed_messages() %}<p class="msg">{{ m }}</p>{% endfor %}
{{ body|safe }}</body></html>"""

def page(tpl, **ctx):
    return render_template_string(BASE, body=render_template_string(tpl, **ctx))

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
            flash("Fill all fields. Password needs at least 6 characters.")
        else:
            try:
                db().execute("INSERT INTO users(name,email,password) VALUES(?,?,?)",
                             (name, email, generate_password_hash(pw)))
                db().commit()
                flash("Registered. Please log in.")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("This email is already registered.")
    return page("""<h3>Register</h3><form method="post">
    <input name="name" placeholder="Name"><input name="email" type="email" placeholder="Email">
    <input name="password" type="password" placeholder="Password (min 6)">
    <button>Register</button></form>""")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = db().execute("SELECT * FROM users WHERE email=?",
                         (request.form["email"].strip().lower(),)).fetchone()
        if u and check_password_hash(u["password"], request.form["password"]):
            session["uid"], session["name"] = u["id"], u["name"]
            return redirect(url_for("upload"))
        flash("Wrong email or password.")
    return page("""<h3>Login</h3><form method="post">
    <input name="email" type="email" placeholder="Email">
    <input name="password" type="password" placeholder="Password">
    <button>Login</button></form>""")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

def make_feedback(target, pred, conf):
    if conf < 0.5:
        return ("I am not sure about this photo. Retake it with both feet "
                "clearly visible, good light and the feet in the lower part of the frame.")
    if target and target == pred:
        return "Good! Your pose matches " + target + ". " + TIPS[pred]
    if target:
        return ("You are practicing " + target + ", but it looks like " + pred +
                ". To correct it: " + TIPS[target])
    return "Detected " + pred + ". " + TIPS[pred]

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        f = request.files.get("photo")
        ext = os.path.splitext(f.filename)[1].lower() if f else ""
        if not f or ext not in {".jpg", ".jpeg", ".png"}:
            flash("Please choose a JPG or PNG photo.")
        else:
            name = uuid.uuid4().hex + ext
            path = os.path.join(UPLOAD_DIR, name)
            f.save(path)
            pred, conf = predict_pada(path)
            pred = str(pred)
            target = request.form.get("target") or None
            fb = make_feedback(target, pred, conf)
            cur = db().execute(
                "INSERT INTO results(user_id,filename,target,predicted,confidence,feedback) "
                "VALUES(?,?,?,?,?,?)", (session["uid"], name, target, pred, conf, fb))
            db().commit()
            return redirect(url_for("result", rid=cur.lastrowid))
    rows = db().execute("SELECT * FROM results WHERE user_id=? ORDER BY id DESC LIMIT 5",
                        (session["uid"],)).fetchall()
    return page("""<h3>Hello, {{ session.name }}</h3>
    <form method="post" enctype="multipart/form-data">
    <label>Which Pada Bheda are you practicing? (optional)</label>
    <select name="target"><option value="">Not sure</option>
    {% for c in classes %}<option>{{ c }}</option>{% endfor %}</select>
    <input type="file" name="photo" accept="image/*" required>
    <button>Check my Pada Bheda</button></form>
    <h4>Recent</h4>
    {% for r in rows %}<div class="card"><a href="/result/{{ r.id }}">{{ r.predicted }}
    ({{ (r.confidence*100)|round|int }}%)</a> - {{ r.created }}</div>{% endfor %}""",
                classes=list(TIPS), rows=rows)

@app.route("/result/<int:rid>")
@login_required
def result(rid):
    r = db().execute("SELECT * FROM results WHERE id=? AND user_id=?",
                     (rid, session["uid"])).fetchone()
    if not r:
        return redirect(url_for("upload"))
    return page("""<h3>Result</h3><img src="/uploads/{{ r.filename }}">
    <div class="card"><b>Pada Bheda:</b> {{ r.predicted }}<br>
    <b>Confidence:</b> {{ (r.confidence*100)|round|int }}%</div>
    <div class="card"><b>Correction feedback</b><br>{{ r.feedback }}</div>
    <a href="/upload">Try another photo</a>""", r=r)

@app.route("/uploads/<fn>")
@login_required
def uploads(fn):
    return send_from_directory(UPLOAD_DIR, fn)

if __name__ == "__main__":
    app.run(debug=True)