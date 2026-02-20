import streamlit as st
import calendar
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import time
import os
import json
import hashlib
import re
from pathlib import Path
import streamlit.components.v1 as components

# =========================
# CONFIG / DEBUG
# =========================
DEBUG = False
try:
    DEBUG = bool(st.secrets.get("DEBUG", False))
except Exception:
    DEBUG = False

def dbg(msg: str):
    if DEBUG:
        st.caption(msg)

start_total = time.time()

st.set_page_config(
    page_title="Gym BRO",
    page_icon="images/gymbro_icon.png",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = str(BASE_DIR / "fitness.db")

# =========================
# CONSTANTS
# =========================
EXERCISE_TYPE = {
    # heavy
    "Bench Press": "heavy",
    "Squat": "heavy",
    "Deadlift": "heavy",
    "Romanian Deadlift": "heavy",
    "Overhead Press": "heavy",
    "Leg Press": "heavy",
    "Barbell Row": "heavy",
    "Lat Pulldown": "heavy",
    "Seated Cable Row": "heavy",

    # light
    "Incline Dumbbell Press": "light",
    "Dumbbell Bench": "light",
    "Dumbbell Flyes": "light",
    "Flat Dumbbell Flyes": "light",
    "Lunges": "light",
    "Leg Curl": "light",
    "Leg Extension": "light",
    "Push-Ups": "light",
    "Pull-Ups": "light",
    "Crunches": "light",
    "Hyperextension": "light",

    # timed
    "Plank": "timed",

    # keep as exercises
    "Biceps": "light",
    "Triceps": "light",
}

TYPE_PROFILES = {
    "heavy": {
        "mode": "weight_reps",
        "weight_options": list(range(0, 301, 5)),
        "reps_options": list(range(0, 51)),
    },
    "light": {
        "mode": "weight_reps",
        "weight_options": sorted(set(
            list(range(0, 11, 1)) +
            list(range(10, 51, 2)) +
            list(range(50, 151, 5))
        )),
        "reps_options": list(range(0, 51)),
    },
    "timed": {
        "mode": "time",
        "time_options": list(range(0, 251, 5)),
    },
}

FAVORITE_EXERCISES = ["Bench Press", "Squat", "Deadlift"]

EXERCISE_IMAGES = {
    "Bench Press": "images/bench.png",
    "Squat": "images/squat.png",
    "Deadlift": "images/deadlift.png",
    "Overhead Press": "images/ohp.png",
    "Biceps": "images/biceps.png",
    "Triceps": "images/triceps.png",
    "Dumbbell Flyes": "images/dumbbell_flyes.png",
    "Romanian Deadlift": "images/romanian_deadlift.png",
    "Incline Dumbbell Press": "images/incline_dumbbell_press.png",
    "Lat Pulldown": "images/lat_pulldown.png",
    "Seated Cable Row": "images/seated_cable_row.png",
    "Dumbbell Bench": "images/dumbbell_bench.png",
    "Push-Ups": "images/push_ups.png",
    "Leg Press": "images/leg_press.png",
    "Lunges": "images/lunges.png",
    "Leg Curl": "images/leg_curl.png",
    "Leg Extension": "images/leg_extension.png",
    "Barbell Row": "images/barbell_row.png",
    "Pull-Ups": "images/pull_ups.png",
    "Plank": "images/plank.png",
    "Crunches": "images/crunches.png",
    "Flat Dumbbell Flyes": "images/flat_dumbbell_flyes.png",
    "Hyperextension": "images/hyperextension.png",
}

SEED_EXERCISES = [
    "Bench Press","Squat","Deadlift","Biceps","Triceps","Overhead Press",
    "Dumbbell Flyes","Romanian Deadlift","Incline Dumbbell Press","Lat Pulldown",
    "Seated Cable Row","Dumbbell Bench","Push-Ups","Leg Press","Lunges","Leg Curl",
    "Leg Extension","Barbell Row","Plank","Pull-Ups","Crunches","Flat Dumbbell Flyes",
    "Hyperextension"
]

# =========================
# STYLES (mobile-friendly chips)
# =========================
st.markdown("""
<style>
@media (max-width: 768px) {
    h1 { font-size: 32px !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 40px !important; }
    .stTabs [data-baseweb="tab"] { font-size: 16px !important; padding: 12px 0px !important; }
    img { max-width: 100% !important; height: auto !important; }
    .block-container { padding: 1rem 1rem 2rem 1rem !important; }
    .set-chip { display:block !important; width: 100% !important; margin: 6px 0 !important; font-size: 18px !important; }
}

.stTabs [data-baseweb="tab-list"] {
    width: 100%;
    justify-content: center;
    gap: 100px;
}
.stTabs [data-baseweb="tab"] {
    font-size: 24px;
    padding: 20px 0px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] { font-weight: 700; }
.stTabs [data-baseweb="tab"]:hover { opacity: 0.8; }

.sets-wrap { margin-top: 6px; }
.set-chip {
    display:inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.14);
    margin: 4px 6px 0 0;
    font-size: 16px;
    line-height: 1.25;
    white-space: nowrap;
}
.set-chip strong { font-weight: 800; }
.small-muted { opacity: .75; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# =========================
# SAFE IMAGE
# =========================
def safe_image(rel_path: str, width: int | None = None):
    if not rel_path:
        return
    p = (BASE_DIR / rel_path).resolve()
    # Don't crash if missing
    if not p.exists() or not p.is_file():
        return
    try:
        st.image(str(p), width=width)
    except Exception:
        # Absolute no-crash policy
        return
    
def read_sets_from_widgets(ns: str, sets_count: int, mode: str) -> list[dict]:
    rows = []
    if mode == "time":
        for i in range(1, sets_count + 1):
            t = int(st.session_state.get(f"{ns}_t_{i}", 0))
            rows.append({"time_sec": t})
    else:
        for i in range(1, sets_count + 1):
            w = int(st.session_state.get(f"{ns}_w_{i}", 0))
            r = int(st.session_state.get(f"{ns}_r_{i}", 0))
            rows.append({"weight": w, "reps": r})
    return rows

# =========================
# DB HELPERS (SQLite only for max speed)
# =========================
@st.cache_resource
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def init_db(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS exercises (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS workouts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workout_date TEXT NOT NULL,
        exercise_id INTEGER NOT NULL,
        FOREIGN KEY (exercise_id) REFERENCES exercises(id)
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workout_id INTEGER NOT NULL,
        set_no INTEGER NOT NULL,
        weight REAL NOT NULL,
        reps INTEGER NOT NULL,
        time_sec INTEGER,
        FOREIGN KEY (workout_id) REFERENCES workouts(id)
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_workouts_date ON workouts(workout_date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_workouts_exercise ON workouts(exercise_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sets_workout ON sets(workout_id)")
    conn.commit()

def _seed_hash(names: list[str]) -> str:
    normalized = sorted([n.strip() for n in names if n and n.strip()])
    payload = json.dumps(normalized, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def seed_exercises_hashed(conn: sqlite3.Connection, names: list[str]):
    """
    Seed only when the seed list changes (per session).
    This prevents doing any seed work on each rerun.
    """
    h = _seed_hash(names)
    key = "_seed_hash_exercises"

    if st.session_state.get(key) == h:
        return

    # Batch insert in one executemany
    clean = [(n.strip(),) for n in names if n and n.strip()]
    if clean:
        cur = conn.cursor()
        cur.executemany("INSERT OR IGNORE INTO exercises(name) VALUES(?)", clean)
        conn.commit()

    st.session_state[key] = h
    get_exercises_df.clear()

# =========================
# CACHED READS (FAST, SMALL)
# =========================
@st.cache_data(ttl=60)
def get_exercises_df() -> pd.DataFrame:
    conn = get_conn()
    return pd.read_sql_query("SELECT id, name FROM exercises ORDER BY name", conn)

@st.cache_data(ttl=30)
def get_month_daily_df(year: int, month: int) -> pd.DataFrame:
    conn = get_conn()
    ym_start = f"{year:04d}-{month:02d}-01"
    last_day = calendar.monthrange(year, month)[1]
    ym_end = f"{year:04d}-{month:02d}-{last_day:02d}"
    return pd.read_sql_query("""
        SELECT workout_date, COUNT(*) AS entries
        FROM workouts
        WHERE workout_date BETWEEN ? AND ?
        GROUP BY workout_date
    """, conn, params=(ym_start, ym_end))

@st.cache_data(ttl=30)
def get_last_workout_compact(exercise_name: str) -> dict | None:
    """
    Returns last workout for exercise as {date:..., sets_text:...}
    Compact in SQL (no big dataframe).
    """
    conn = get_conn()
    row = conn.execute("""
        SELECT w.id
        FROM workouts w
        JOIN exercises e ON e.id = w.exercise_id
        WHERE e.name = ?
        ORDER BY w.workout_date DESC, w.id DESC
        LIMIT 1
    """, (exercise_name,)).fetchone()

    if not row:
        return None

    workout_id = int(row[0])

    # ordered sets -> group_concat
    df = pd.read_sql_query("""
        SELECT
            w.workout_date AS workout_date,
            s.set_no,
            CASE
              WHEN s.time_sec IS NOT NULL AND s.time_sec > 0 THEN printf('%ds', s.time_sec)
              ELSE printf('%d×%d', CAST(s.weight AS INT), s.reps)
            END AS set_str
        FROM workouts w
        JOIN sets s ON s.workout_id = w.id
        WHERE w.id = ?
        ORDER BY s.set_no ASC
    """, conn, params=(workout_id,))

    if df.empty:
        return None

    d = str(df.iloc[0]["workout_date"])
    sets_text = " | ".join(df["set_str"].tolist())
    return {"workout_id": workout_id, "date": d, "sets_text": sets_text}

@st.cache_data(ttl=20)
def get_history_compact(exercise: str | None, d_from: str, d_to: str) -> pd.DataFrame:
    """
    1 row = 1 workout entry (date+exercise+workout_id) with sets already aggregated in SQL.
    """
    conn = get_conn()

    # Use subquery for ordered group_concat
    base = """
        SELECT
            t.workout_id,
            t.workout_date,
            t.exercise,
            group_concat(t.set_str, ' | ') AS sets
        FROM (
            SELECT
                w.id AS workout_id,
                w.workout_date,
                e.name AS exercise,
                s.set_no,
                CASE
                  WHEN s.time_sec IS NOT NULL AND s.time_sec > 0 THEN printf('%ds', s.time_sec)
                  ELSE printf('%d×%d', CAST(s.weight AS INT), s.reps)
                END AS set_str
            FROM workouts w
            JOIN exercises e ON e.id = w.exercise_id
            JOIN sets s ON s.workout_id = w.id
            WHERE w.workout_date BETWEEN ? AND ?
            {ex_filter}
            ORDER BY w.workout_date DESC, w.id DESC, s.set_no ASC
        ) t
        GROUP BY t.workout_id, t.workout_date, t.exercise
        ORDER BY t.workout_date DESC, t.workout_id DESC
    """

    if exercise and exercise != "All":
        q = base.format(ex_filter="AND e.name = ?")
        return pd.read_sql_query(q, conn, params=(d_from, d_to, exercise))

    q = base.format(ex_filter="")
    return pd.read_sql_query(q, conn, params=(d_from, d_to))

@st.cache_data(ttl=20)
def get_exercise_sets_for_progress(exercise_name: str) -> pd.DataFrame:
    """
    Only rows needed for progress chart (not the whole history).
    """
    conn = get_conn()
    return pd.read_sql_query("""
        SELECT
            w.workout_date,
            s.weight,
            s.reps
        FROM workouts w
        JOIN exercises e ON e.id = w.exercise_id
        JOIN sets s ON s.workout_id = w.id
        WHERE e.name = ?
          AND (s.time_sec IS NULL OR s.time_sec = 0)
          AND s.weight > 0 AND s.reps > 0
        ORDER BY w.workout_date ASC, w.id ASC, s.set_no ASC
    """, conn, params=(exercise_name,))

def clear_cache_after_write():
    get_exercises_df.clear()
    get_month_daily_df.clear()
    get_last_workout_compact.clear()
    get_history_compact.clear()
    get_exercise_sets_for_progress.clear()

# =========================
# WRITES
# =========================
def upsert_exercise(conn: sqlite3.Connection, name: str) -> int:
    name = name.strip()
    if not name:
        raise ValueError("Empty exercise name")
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO exercises(name) VALUES(?)", (name,))
    conn.commit()
    row = cur.execute("SELECT id FROM exercises WHERE name = ?", (name,)).fetchone()
    if not row:
        raise RuntimeError("Failed to fetch exercise id")
    return int(row[0])

def insert_workout(conn: sqlite3.Connection, workout_date: str, exercise_id: int, sets_rows: list[dict]):
    cur = conn.cursor()
    cur.execute("INSERT INTO workouts(workout_date, exercise_id) VALUES(?, ?)", (workout_date, exercise_id))
    workout_id = int(cur.lastrowid)

    payload = []
    for i, s in enumerate(sets_rows, start=1):
        payload.append((
            workout_id,
            i,
            float(s.get("weight", 0)),
            int(s.get("reps", 0)),
            int(s["time_sec"]) if s.get("time_sec") is not None else None
        ))

    cur.executemany("""
        INSERT INTO sets(workout_id, set_no, weight, reps, time_sec)
        VALUES(?, ?, ?, ?, ?)
    """, payload)
    conn.commit()
    clear_cache_after_write()

def delete_workout(conn: sqlite3.Connection, workout_id: int):
    cur = conn.cursor()
    cur.execute("DELETE FROM sets WHERE workout_id = ?", (workout_id,))
    cur.execute("DELETE FROM workouts WHERE id = ?", (workout_id,))
    conn.commit()
    clear_cache_after_write()

def update_workout_full(conn: sqlite3.Connection, workout_id: int, workout_date: str, exercise_id: int, sets_rows: list[dict]):
    """
    Replace workout: update workouts(date, exercise_id), replace all sets.
    """
    if not sets_rows:
        raise ValueError("No sets to save")

    cur = conn.cursor()
    cur.execute("UPDATE workouts SET workout_date = ?, exercise_id = ? WHERE id = ?", (workout_date, exercise_id, workout_id))

    cur.execute("DELETE FROM sets WHERE workout_id = ?", (workout_id,))
    payload = []
    for i, s in enumerate(sets_rows, start=1):
        payload.append((
            workout_id,
            i,
            float(s.get("weight", 0)),
            int(s.get("reps", 0)),
            int(s["time_sec"]) if s.get("time_sec") is not None else None
        ))
    cur.executemany("""
        INSERT INTO sets(workout_id, set_no, weight, reps, time_sec)
        VALUES(?, ?, ?, ?, ?)
    """, payload)

    conn.commit()
    clear_cache_after_write()

def get_workout_for_edit(conn: sqlite3.Connection, workout_id: int) -> dict:
    row = conn.execute("""
        SELECT w.id, w.workout_date, e.name as exercise
        FROM workouts w
        JOIN exercises e ON e.id = w.exercise_id
        WHERE w.id = ?
    """, (workout_id,)).fetchone()
    if not row:
        raise ValueError("Workout not found")

    sets = pd.read_sql_query("""
        SELECT set_no, weight, reps, time_sec
        FROM sets
        WHERE workout_id = ?
        ORDER BY set_no ASC
    """, conn, params=(workout_id,))

    return {
        "workout_id": int(row[0]),
        "workout_date": str(row[1]),
        "exercise": str(row[2]),
        "sets": sets
    }

# =========================
# COPY TO CLIPBOARD (JS)
# =========================
def copy_to_clipboard_button(text: str, label: str = "📋 Copy to clipboard", key: str = "copy_btn"):
    # Escape for JS string
    safe = text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    html = f"""
    <div style="margin: 8px 0;">
      <button id="{key}" style="
        padding: 10px 14px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.2);
        background: rgba(255,255,255,0.08); color: white; cursor: pointer; font-size: 14px;">
        {label}
      </button>
      <span id="{key}_status" style="margin-left:10px; opacity:.8; font-size: 13px;"></span>
    </div>
    <script>
      const btn = document.getElementById("{key}");
      const status = document.getElementById("{key}_status");
      btn.onclick = async () => {{
        try {{
          await navigator.clipboard.writeText(`{safe}`);
          status.textContent = "Copied ✅";
          setTimeout(() => status.textContent = "", 1200);
        }} catch (e) {{
          status.textContent = "Copy failed (browser blocked)";
          setTimeout(() => status.textContent = "", 1800);
        }}
      }};
    </script>
    """
    components.html(html, height=60)

# =========================
# HEADER
# =========================
col1, col2 = st.columns([1, 6])
with col1:
    safe_image("images/gymbro_logo.png", width=90)
with col2:
    st.markdown("<h1 style='margin:0'>Gym BRO</h1>", unsafe_allow_html=True)

# =========================
# INIT DB + SEED (FAST)
# =========================
conn = get_conn()
init_db(conn)
seed_exercises_hashed(conn, SEED_EXERCISES)

tab_add, tab_history, tab_progress = st.tabs(["➕ Add workout", "📜 History", "📈 Progress"])

# ============================
# TAB: Add workout
# ============================
with tab_add:
    st.subheader("Add workout")

    workout_date = st.date_input(
        "📅 Date",
        value=date.today(),
        min_value=date(2020, 1, 1),
        max_value=date.today(),
        key="workout_date"
    )

    ex_df = get_exercises_df()
    ex_names = ex_df["name"].tolist()

    q = st.text_input("Search exercise", "", key="search_ex")
    filtered = [x for x in ex_names if q.lower() in x.lower()] if q else ex_names
    fav = [x for x in FAVORITE_EXERCISES if x in filtered]
    rest = [x for x in filtered if x not in fav]
    ex_options = fav + rest

    exercise_name = st.selectbox(
        "Exercise",
        ex_options,
        key="add_exercise_select"
    )

    img_path = EXERCISE_IMAGES.get(exercise_name)

    if img_path and os.path.exists(img_path):
        st.image(img_path, width=140)

    ns = re.sub(r"[^a-zA-Z0-9_]+", "_", f"{workout_date}_{exercise_name}".replace(" ", "_"))
    sets_key = f"sets_{ns}"

    ex_type = EXERCISE_TYPE.get(exercise_name, "light")
    profile = TYPE_PROFILES[ex_type]

    if sets_key not in st.session_state:
        st.session_state[sets_key] = [{"time_sec": 0}] if profile["mode"] == "time" else [{"weight": 0, "reps": 0}]

# ---------- Sets form ----------
    sets_rows: list[dict] = []

    with st.form(f"sets_form_{ns}", clear_on_submit=False):
        for idx, s in enumerate(st.session_state[sets_key], start=1):

            if profile["mode"] == "time":
                key_t = f"{ns}_t_{idx}"
                current_t = int(st.session_state.get(key_t, s.get("time_sec", 0) or 0))

                t = st.selectbox(
                    f"Set {idx} — Time (sec)",
                    profile["time_options"],
                    index=profile["time_options"].index(current_t) if current_t in profile["time_options"] else 0,
                    key=key_t
                )
                sets_rows.append({"time_sec": int(t)})

            else:
                c1, c2 = st.columns(2)

                key_w = f"{ns}_w_{idx}"
                key_r = f"{ns}_r_{idx}"

                current_w = int(st.session_state.get(key_w, s.get("weight", 0) or 0))
                current_r = int(st.session_state.get(key_r, s.get("reps", 0) or 0))

                with c1:
                    w = st.selectbox(
                        f"Set {idx} — Weight (kg)",
                        profile["weight_options"],
                        index=profile["weight_options"].index(current_w) if current_w in profile["weight_options"] else 0,
                        key=key_w
                    )
                with c2:
                    r = st.selectbox(
                        f"Set {idx} — Reps",
                        profile["reps_options"],
                        index=profile["reps_options"].index(current_r) if current_r in profile["reps_options"] else 0,
                        key=key_r
                    )

                sets_rows.append({"weight": int(w), "reps": int(r)})

        apply = st.form_submit_button("✅ Apply sets")

# Apply updates (optional, for instant summary refresh)
if apply:
    st.session_state[sets_key] = sets_rows
    st.rerun()

# ---------- Session summary ----------
st.markdown("### Session summary")

# show summary based on current widgets (so it matches what's on screen)
current_sets = read_sets_from_widgets(ns, len(st.session_state[sets_key]), profile["mode"])
st.session_state[sets_key] = current_sets  # keep state consistent

if profile["mode"] == "time":
    filled = [s for s in current_sets if s.get("time_sec", 0) > 0]
    total_t = sum(s["time_sec"] for s in filled) if filled else 0
    st.info(f"Sets: {len(filled)} | Total time: {total_t} sec")
    chips = "".join(
        [f'<span class="set-chip"><strong>{i+1}</strong> · {s["time_sec"]}s</span>'
         for i, s in enumerate(filled)]
    )
else:
    filled = [s for s in current_sets if s.get("weight", 0) > 0 and s.get("reps", 0) > 0]
    vol = sum(s["weight"] * s["reps"] for s in filled) if filled else 0
    st.info(f"Sets: {len(filled)} | Total volume: {vol} kg")
    chips = "".join(
        [f'<span class="set-chip"><strong>{i+1}</strong> · {s["weight"]}×{s["reps"]}</span>'
         for i, s in enumerate(filled)]
    )

st.markdown(f'<div class="sets-wrap">{chips}</div>', unsafe_allow_html=True)

# ---------- Save workout (outside form) ----------
if st.button("💾 Save workout", key=f"{ns}_save_btn"):
    try:
        # read exact on-screen values
        current_sets = read_sets_from_widgets(ns, len(st.session_state[sets_key]), profile["mode"])
        st.session_state[sets_key] = current_sets

        if profile["mode"] == "time":
            cleaned = [s for s in current_sets if s.get("time_sec", 0) > 0]
            normalized = [{"weight": 0, "reps": 0, "time_sec": int(s["time_sec"])} for s in cleaned]
        else:
            cleaned = [s for s in current_sets if s.get("weight", 0) > 0 and s.get("reps", 0) > 0]
            normalized = [{"weight": int(s["weight"]), "reps": int(s["reps"]), "time_sec": None} for s in cleaned]

        if not normalized:
            st.error("Add at least one filled set.")
            st.stop()

        ex_id = upsert_exercise(conn, exercise_name)
        insert_workout(conn, str(workout_date), ex_id, normalized)

        st.success("Saved ✅")

        # reset sets for this namespace
        st.session_state[sets_key] = [{"time_sec": 0}] if profile["mode"] == "time" else [{"weight": 0, "reps": 0}]

        # clear widget keys
        for i in range(1, 60):
            st.session_state.pop(f"{ns}_w_{i}", None)
            st.session_state.pop(f"{ns}_r_{i}", None)
            st.session_state.pop(f"{ns}_t_{i}", None)

        st.rerun()

    except Exception as e:
        st.error(f"Save failed: {e}")

def read_sets_from_widgets(ns: str, sets_count: int, mode: str) -> list[dict]:
    rows = []
    if mode == "time":
        for i in range(1, sets_count + 1):
            t = int(st.session_state.get(f"{ns}_t_{i}", 0))
            rows.append({"time_sec": t})
    else:
        for i in range(1, sets_count + 1):
            w = int(st.session_state.get(f"{ns}_w_{i}", 0))
            r = int(st.session_state.get(f"{ns}_r_{i}", 0))
            rows.append({"weight": w, "reps": r})
    return rows

# ============================
# TAB: History (FAST + COPY + EDIT)
# ============================
with tab_history:
    st.subheader("History")

    # Filters first (cheap)
    ex_df = get_exercises_df()
    ex_list = ["All"] + ex_df["name"].tolist()
    c1, c2, c3 = st.columns([2, 2, 2])

    with c1:
        ex_filter = st.selectbox("Exercise", ex_list, index=0, key="hist_ex_filter")

    # date boundaries from DB quickly
    row = conn.execute("SELECT MIN(workout_date), MAX(workout_date) FROM workouts").fetchone()
    if not row or row[0] is None:
        st.info("No workouts yet.")
        st.stop()

    dmin = pd.to_datetime(row[0]).date()
    dmax = pd.to_datetime(row[1]).date()

    with c2:
        d_from = st.date_input("From", value=dmin, min_value=dmin, max_value=dmax, key="hist_from")
    with c3:
        d_to = st.date_input("To", value=dmax, min_value=dmin, max_value=dmax, key="hist_to")

    t0 = time.time()
    view = get_history_compact(ex_filter, str(d_from), str(d_to))
    dbg(f"history_compact: {time.time() - t0:.3f}s | rows={len(view)}")

    if view.empty:
        st.info("No records for current filters.")
        st.stop()

    # Copy history text
    lines = []
    for _, r in view.iterrows():
        lines.append(f"{r['workout_date']} — {r['exercise']}: {r['sets']}")
    history_text = "\n".join(lines)

    st.markdown("### Export / Copy")
    copy_to_clipboard_button(history_text, key="copy_history_btn")

    st.download_button(
        "⬇️ Download history (.txt)",
        data=history_text.encode("utf-8"),
        file_name="gymbro_history.txt",
        mime="text/plain"
    )

    st.markdown("---")

    # Render day groups
    # Grouping in pandas is fine because view is already compact & small
    for day, day_df in view.groupby("workout_date", sort=False):
        with st.expander(f"📅 {day}  ·  {len(day_df)} entries", expanded=True):
            day_df = day_df.sort_values("workout_id", ascending=False)

            for _, r in day_df.iterrows():
                workout_id = int(r["workout_id"])

                left, mid, right = st.columns([6, 2, 2])

                with left:
                    st.markdown(f"**{r['exercise']}**")
                    # show as chips
                    chips = "".join([f'<span class="set-chip">{s.strip()}</span>' for s in str(r["sets"]).split("|")])
                    st.markdown(f'<div class="sets-wrap">{chips}</div>', unsafe_allow_html=True)

                with mid:
                    pop = st.popover("✏️ Edit", use_container_width=True)
                    with pop:
                        try:
                            data = get_workout_for_edit(conn, workout_id)
                        except Exception as e:
                            st.error(str(e))
                            st.stop()

                        st.markdown("**Edit workout**")
                        new_date = st.date_input(
                            "Date",
                            value=pd.to_datetime(data["workout_date"]).date(),
                            key=f"edit_date_{workout_id}"
                        )

                        ex_df2 = get_exercises_df()
                        ex_names2 = ex_df2["name"].tolist()
                        new_ex = st.selectbox(
                            "Exercise",
                            ex_names2,
                            index=ex_names2.index(data["exercise"]) if data["exercise"] in ex_names2 else 0,
                            key=f"edit_ex_{workout_id}"
                        )

                        ex_type = EXERCISE_TYPE.get(new_ex, "light")
                        profile = TYPE_PROFILES[ex_type]
                        mode = profile["mode"]

                        # sets editor
                        current_sets = data["sets"].to_dict("records")
                        # normalize current sets into mode
                        if mode == "time":
                            # if old was weight sets, convert to time=0
                            norm = []
                            for s in current_sets:
                                t = int(s["time_sec"]) if pd.notna(s["time_sec"]) else 0
                                norm.append({"time_sec": t})
                            current_sets = norm if norm else [{"time_sec": 0}]
                        else:
                            norm = []
                            for s in current_sets:
                                w = int(s["weight"]) if pd.notna(s["weight"]) else 0
                                rps = int(s["reps"]) if pd.notna(s["reps"]) else 0
                                norm.append({"weight": w, "reps": rps})
                            current_sets = norm if norm else [{"weight": 0, "reps": 0}]

                        cnt = st.number_input(
                            "Sets count",
                            min_value=1,
                            max_value=20,
                            value=len(current_sets),
                            step=1,
                            key=f"edit_cnt_{workout_id}"
                        )

                        # resize list to cnt
                        while len(current_sets) < cnt:
                            current_sets.append({"time_sec": 0} if mode == "time" else {"weight": 0, "reps": 0})
                        while len(current_sets) > cnt:
                            current_sets.pop()

                        edited_rows = []
                        if mode == "time":
                            for i in range(cnt):
                                t = st.selectbox(
                                    f"Set {i+1} — Time (sec)",
                                    profile["time_options"],
                                    index=profile["time_options"].index(current_sets[i]["time_sec"]) if current_sets[i]["time_sec"] in profile["time_options"] else 0,
                                    key=f"edit_t_{workout_id}_{i}"
                                )
                                edited_rows.append({"weight": 0, "reps": 0, "time_sec": int(t)})
                        else:
                            for i in range(cnt):
                                cA, cB = st.columns(2)
                                with cA:
                                    w = st.selectbox(
                                        f"Set {i+1} — Weight (kg)",
                                        profile["weight_options"],
                                        index=profile["weight_options"].index(current_sets[i]["weight"]) if current_sets[i]["weight"] in profile["weight_options"] else 0,
                                        key=f"edit_w_{workout_id}_{i}"
                                    )
                                with cB:
                                    rr = st.selectbox(
                                        f"Set {i+1} — Reps",
                                        profile["reps_options"],
                                        index=profile["reps_options"].index(current_sets[i]["reps"]) if current_sets[i]["reps"] in profile["reps_options"] else 0,
                                        key=f"edit_r_{workout_id}_{i}"
                                    )
                                edited_rows.append({"weight": int(w), "reps": int(rr), "time_sec": None})

                        # Validate: at least one filled set
                        if mode == "time":
                            cleaned = [s for s in edited_rows if s.get("time_sec", 0) > 0]
                        else:
                            cleaned = [s for s in edited_rows if s.get("weight", 0) > 0 and s.get("reps", 0) > 0]

                        if st.button("💾 Save changes", key=f"edit_save_{workout_id}"):
                            if not cleaned:
                                st.error("Fill at least one set.")
                            else:
                                ex_id = upsert_exercise(conn, new_ex)
                                update_workout_full(conn, workout_id, str(new_date), ex_id, cleaned)
                                st.success("Updated ✅")
                                st.rerun()

                with right:
                    pop = st.popover("🗑 Delete", use_container_width=True)
                    with pop:
                        st.write("Delete this entry?")
                        confirm = st.checkbox("Confirm", key=f"confirm_{workout_id}")
                        if st.button("Delete", key=f"del_{workout_id}", disabled=not confirm):
                            delete_workout(conn, workout_id)
                            st.success("Deleted ✅")
                            st.rerun()

# ============================
# TAB: Progress (FAST)
# ============================
with tab_progress:
    st.subheader("Progress")

    st.markdown("## 🗓 Training calendar")

    if "cal_month" not in st.session_state or "cal_year" not in st.session_state:
        st.session_state.cal_year = date.today().year
        st.session_state.cal_month = date.today().month

    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if st.button("◀", key="cal_prev"):
            m = st.session_state.cal_month - 1
            y = st.session_state.cal_year
            if m == 0:
                m = 12
                y -= 1
            st.session_state.cal_month, st.session_state.cal_year = m, y
            st.rerun()

    with c2:
        month_name = calendar.month_name[st.session_state.cal_month]
        st.markdown(f"### {month_name} {st.session_state.cal_year}")

    with c3:
        if st.button("▶", key="cal_next"):
            m = st.session_state.cal_month + 1
            y = st.session_state.cal_year
            if m == 13:
                m = 1
                y += 1
            st.session_state.cal_month, st.session_state.cal_year = m, y
            st.rerun()

    t_cal = time.time()
    month_daily = get_month_daily_df(st.session_state.cal_year, st.session_state.cal_month)
    dbg(f"month_daily: {time.time() - t_cal:.3f}s | rows={len(month_daily)}")

    entries_map = {}
    if not month_daily.empty:
        for _, r in month_daily.iterrows():
            d = pd.to_datetime(r["workout_date"]).date().day
            entries_map[int(d)] = int(r["entries"])

    calobj = calendar.Calendar(firstweekday=0)
    weeks = calobj.monthdayscalendar(st.session_state.cal_year, st.session_state.cal_month)

    css = """
    <style>
    .cal-wrap { width: 100%; max-width: 720px; }
    .cal-grid { width: 100%; border-collapse: separate; border-spacing: 8px; }
    .cal-grid th { text-align: center; font-size: 14px; opacity: 0.8; padding-bottom: 4px; }
    .cal-grid td { height: 52px; border-radius: 12px; text-align: center; vertical-align: middle; font-size: 16px; }
    .cal-day { background: rgba(255,255,255,0.06); }
    .cal-empty { background: transparent; }
    .cal-trained { background: rgba(0, 200, 83, 0.28); border: 1px solid rgba(0, 200, 83, 0.45); }
    .cal-count { display:block; font-size: 12px; opacity: 0.85; margin-top: 2px; }
    </style>
    """

    headers = ["Пн","Вт","Ср","Чт","Пт","Сб","Вс"]
    html = [css, '<div class="cal-wrap"><table class="cal-grid"><thead><tr>']
    html += [f"<th>{h}</th>" for h in headers]
    html.append("</tr></thead><tbody>")

    for w in weeks:
        html.append("<tr>")
        for d in w:
            if d == 0:
                html.append('<td class="cal-empty"></td>')
            else:
                trained = d in entries_map
                cls = "cal-day cal-trained" if trained else "cal-day"
                cnt = f'<span class="cal-count">{entries_map[d]} записи</span>' if trained else '<span class="cal-count">&nbsp;</span>'
                html.append(f'<td class="{cls}">{d}{cnt}</td>')
        html.append("</tr>")

    html.append("</tbody></table></div>")
    st.markdown("".join(html), unsafe_allow_html=True)

    st.divider()

    st.markdown("## 📈 Exercise progress")

    ex_df = get_exercises_df()
    ex_names = ex_df["name"].tolist()
    if not ex_names:
        st.info("No exercises.")
        st.stop()

    ex = st.selectbox("Exercise", ex_names, key="progress_exercise_select")

    t_p = time.time()
    df = get_exercise_sets_for_progress(ex)
    dbg(f"progress_sets: {time.time() - t_p:.3f}s | rows={len(df)}")

    if df.empty:
        st.info("No weight+reps sets for this exercise yet.")
        st.stop()

    # Estimated 1RM
    df["est_1rm"] = df["weight"] * (1 + (df["reps"] / 30.0))

    best_1rm_by_day = df.groupby("workout_date", as_index=False)["est_1rm"].max()
    top_w_by_day = df.groupby("workout_date", as_index=False)["weight"].max()

    fig, ax1 = plt.subplots()
    ax1.plot(best_1rm_by_day["workout_date"], best_1rm_by_day["est_1rm"])
    ax1.set_title(f"{ex} — Estimated 1RM & Top weight by day")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Estimated 1RM")
    plt.xticks(rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(top_w_by_day["workout_date"], top_w_by_day["weight"])
    ax2.set_ylabel("Top weight (kg)")

    st.pyplot(fig)
    plt.close(fig)

    st.metric("🏆 Best estimated 1RM", f"{float(df['est_1rm'].max()):.1f}")

dbg(f"Render: {time.time() - start_total:.3f} sec")