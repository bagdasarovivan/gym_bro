import streamlit as st
import calendar
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta
import time
import json
import hashlib
import re
import os
import psycopg2
import psycopg2.extras
from pathlib import Path
import streamlit.components.v1 as components
import uuid

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

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = str((BASE_DIR / "fitness.db").resolve())

ICON_PATH = str((BASE_DIR / "images" / "gymbro_icon.png").resolve())
st.set_page_config(page_title="Gym BRO", page_icon=ICON_PATH, layout="centered")

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
        "weight_options": sorted(
            set(list(range(0, 11, 1)) + list(range(10, 51, 2)) + list(range(50, 151, 5)))
        ),
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
    "Bench Press",
    "Squat",
    "Deadlift",
    "Biceps",
    "Triceps",
    "Overhead Press",
    "Dumbbell Flyes",
    "Romanian Deadlift",
    "Incline Dumbbell Press",
    "Lat Pulldown",
    "Seated Cable Row",
    "Dumbbell Bench",
    "Push-Ups",
    "Leg Press",
    "Lunges",
    "Leg Curl",
    "Leg Extension",
    "Barbell Row",
    "Plank",
    "Pull-Ups",
    "Crunches",
    "Flat Dumbbell Flyes",
    "Hyperextension",
]

# =========================
# STYLES
# =========================
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# =========================
# SAFE IMAGE
# =========================
def safe_image(rel_path: str | None, width: int | None = None):
    if not rel_path:
        return
    p = (BASE_DIR / rel_path).resolve()
    if not p.exists() or not p.is_file():
        return
    try:
        st.image(str(p), width=width)
    except Exception:
        return


def read_sets_from_widgets(ns: str, sets_count: int, mode: str) -> list[dict]:
    rows: list[dict] = []
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
# DB HELPERS
# =========================
@st.cache_resource
def get_conn():
    db_url = st.secrets.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("Нет DATABASE_URL в st.secrets (Supabase не подключён)")
    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    return conn

def init_db(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS exercises (
                id BIGSERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS workouts (
                id BIGSERIAL PRIMARY KEY,
                workout_date DATE NOT NULL,
                exercise_id BIGINT NOT NULL REFERENCES exercises(id) ON DELETE CASCADE
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sets (
                id BIGSERIAL PRIMARY KEY,
                workout_id BIGINT NOT NULL REFERENCES workouts(id) ON DELETE CASCADE,
                set_no INT NOT NULL,
                weight DOUBLE PRECISION NOT NULL,
                reps INT NOT NULL,
                time_sec INT
            );
            """
        )

        cur.execute("CREATE INDEX IF NOT EXISTS idx_workouts_date ON workouts(workout_date);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_workouts_exercise ON workouts(exercise_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sets_workout ON sets(workout_id);")

    conn.commit()


def _seed_hash(names: list[str]) -> str:
    normalized = sorted([n.strip() for n in names if n and n.strip()])
    payload = json.dumps(normalized, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def seed_exercises_hashed(conn, names: list[str]):
    """
    Seed only when the seed list changes (per session).
    """
    h = _seed_hash(names)
    key = "_seed_hash_exercises"
    if st.session_state.get(key) == h:
        return

    clean = [(n.strip(),) for n in names if n and n.strip()]
    if clean:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO exercises(name) VALUES(%s) ON CONFLICT (name) DO NOTHING",
                clean,
            )
        conn.commit()
        st.session_state[key] = h
        get_exercises_df.clear()

# =========================
# CACHED READS
# =========================
@st.cache_data(ttl=600)
def get_exercises_df() -> pd.DataFrame:
    conn = get_conn()
    return pd.read_sql_query("SELECT id, name FROM exercises ORDER BY name", conn)


@st.cache_data(ttl=600)
def get_month_daily_df(year: int, month: int) -> pd.DataFrame:
    conn = get_conn()
    ym_start = f"{year:04d}-{month:02d}-01"
    last_day = calendar.monthrange(year, month)[1]
    ym_end = f"{year:04d}-{month:02d}-{last_day:02d}"
    return pd.read_sql_query(
    """
    SELECT workout_date, COUNT(*) AS entries
    FROM workouts
    WHERE workout_date BETWEEN %s AND %s
    GROUP BY workout_date
    """,
    conn,
    params=(ym_start, ym_end),
    )


@st.cache_data(ttl=300)
def get_history_compact(exercise: str | None, d_from: str, d_to: str) -> pd.DataFrame:
    conn = get_conn()

    base = """
        SELECT
            w.id AS workout_id,
            w.workout_date,
            e.name AS exercise,
            string_agg(
                CASE
                    WHEN s.time_sec IS NOT NULL AND s.time_sec > 0 THEN (s.time_sec::text || 's')
                    ELSE (s.weight::int::text || '×' || s.reps::text)
                END,
                ' | '
                ORDER BY s.set_no
            ) AS sets
        FROM workouts w
        JOIN exercises e ON e.id = w.exercise_id
        JOIN sets s ON s.workout_id = w.id
        WHERE w.workout_date BETWEEN %s AND %s
        {ex_filter}
        GROUP BY w.id, w.workout_date, e.name
        ORDER BY w.workout_date DESC, w.id DESC
    """

    if exercise and exercise != "All":
        q = base.format(ex_filter="AND e.name = %s")
        return pd.read_sql_query(q, conn, params=(d_from, d_to, exercise))

    q = base.format(ex_filter="")
    return pd.read_sql_query(q, conn, params=(d_from, d_to))


@st.cache_data(ttl=300)
def get_exercise_sets_for_progress(exercise_name: str) -> pd.DataFrame:
    conn = get_conn()
    return pd.read_sql_query(
        """
        SELECT
            w.workout_date,
            s.weight,
            s.reps
        FROM workouts w
        JOIN exercises e ON e.id = w.exercise_id
        JOIN sets s ON s.workout_id = w.id
        WHERE e.name = %s
        AND (s.time_sec IS NULL OR s.time_sec = 0)
        AND s.weight > 0 AND s.reps > 0
        ORDER BY w.workout_date ASC, w.id ASC, s.set_no ASC
        """,
        conn,
        params=(exercise_name,),
    )


def clear_cache_after_write():
    get_exercises_df.clear()
    get_month_daily_df.clear()
    get_history_compact.clear()
    get_exercise_sets_for_progress.clear()


# =========================
# WRITES
# =========================
def upsert_exercise(conn, name: str) -> int:
    name = name.strip()
    if not name:
        raise ValueError("Empty exercise name")

    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO exercises(name) VALUES(%s) ON CONFLICT (name) DO NOTHING",
            (name,),
        )
        cur.execute("SELECT id FROM exercises WHERE name = %s", (name,))
        row = cur.fetchone()

    conn.commit()

    if not row:
        raise RuntimeError("Failed to fetch exercise id")
    return int(row[0])

def insert_workout(conn, workout_date: str, exercise_id: int, sets_rows: list[dict]):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO workouts(workout_date, exercise_id) VALUES(%s, %s) RETURNING id",
            (workout_date, exercise_id),
        )
        workout_id = int(cur.fetchone()[0])

        payload: list[tuple] = []
        for i, s in enumerate(sets_rows, start=1):
            payload.append(
                (
                    workout_id,
                    i,
                    float(s.get("weight", 0)),
                    int(s.get("reps", 0)),
                    int(s["time_sec"]) if s.get("time_sec") is not None else None,
                )
            )

        cur.executemany(
            """
            INSERT INTO sets(workout_id, set_no, weight, reps, time_sec)
            VALUES(%s, %s, %s, %s, %s)
            """,
            payload,
        )

    conn.commit()
    clear_cache_after_write()

def delete_workout(conn, workout_id: int):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM workouts WHERE id = %s", (workout_id,))
    conn.commit()
    clear_cache_after_write()


def get_workout_for_edit(conn, workout_id: int) -> dict:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT w.id, w.workout_date, e.name
            FROM workouts w
            JOIN exercises e ON e.id = w.exercise_id
            WHERE w.id = %s
        """, (workout_id,))
        row = cur.fetchone()

    if not row:
        raise ValueError("Workout not found")

    sets = pd.read_sql_query("""
        SELECT set_no, weight, reps, time_sec
        FROM sets
        WHERE workout_id = %s
        ORDER BY set_no ASC
    """, conn, params=(workout_id,))

    return {"workout_id": int(row[0]), "workout_date": str(row[1]), "exercise": str(row[2]), "sets": sets}

# =========================
# COPY TO CLIPBOARD (JS)
# =========================
def copy_to_clipboard_button(text: str, label: str = "📋 Copy", key: str = "copy_btn"):
    js_text = json.dumps(text, ensure_ascii=False)
    js_key = json.dumps(key, ensure_ascii=False)
    js_status = json.dumps(key + "_status", ensure_ascii=False)

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
      const btn = document.getElementById({js_key});
      const status = document.getElementById({js_status});

      btn.onclick = async () => {{
        try {{
          await navigator.clipboard.writeText({js_text});
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
# INIT DB + SEED
# =========================
try:
    conn = get_conn()
    init_db(conn)
    seed_exercises_hashed(conn, SEED_EXERCISES)
except Exception as e:
    st.error(f"DB connect failed: {e}")
    st.stop()    

tab_add, tab_history, tab_progress = st.tabs(["➕ Add workout", "📜 History", "📈 Progress"])

# =========================
# TAB: Add workout
# =========================
with tab_add:
    st.subheader("Add workout")

    workout_date = st.date_input(
        "📅 Date",
        value=date.today(),
        min_value=date(2020, 1, 1),
        max_value=date.today(),
        key="workout_date",
    )

    ex_df = get_exercises_df()
    ex_names = ex_df["name"].tolist()

    q = st.text_input("Search exercise", "", key="search_ex")
    filtered = [x for x in ex_names if q.lower() in x.lower()] if q else ex_names
    fav = [x for x in FAVORITE_EXERCISES if x in filtered]
    rest = [x for x in filtered if x not in fav]
    ex_options = fav + rest

    exercise_name = st.selectbox("Exercise", ex_options, key="add_exercise_select")
    safe_image(EXERCISE_IMAGES.get(exercise_name), width=140)

    ex_type = EXERCISE_TYPE.get(exercise_name, "light")
    profile = TYPE_PROFILES[ex_type]
    mode = profile["mode"]

    # Stable form namespace: avoids "garbage" states when date/exercise changes
    # We keep a per-exercise UUID until exercise changes.
    if "add_ns_by_ex" not in st.session_state:
        st.session_state.add_ns_by_ex = {}

    if exercise_name not in st.session_state.add_ns_by_ex:
        st.session_state.add_ns_by_ex[exercise_name] = uuid.uuid4().hex[:10]

    ns = f"add_{st.session_state.add_ns_by_ex[exercise_name]}"
    sets_key = f"sets_{ns}"

    # -------------------------
    # SETS UI (Apply inside box, Add/Remove below)
    # -------------------------

    # init list state
    if sets_key not in st.session_state:
        st.session_state[sets_key] = [{"time_sec": 0}] if mode == "time" else [{"weight": 0, "reps": 0}]

    sets_rows: list[dict] = []

    # "box" area: selects + Apply button inside
    with st.container():
        for idx, s in enumerate(st.session_state[sets_key], start=1):
            if mode == "time":
                key_t = f"{ns}_t_{idx}"

                # init widget state once
                if key_t not in st.session_state:
                    st.session_state[key_t] = int(s.get("time_sec", 0) or 0)

                t = st.selectbox(
                    f"Set {idx} — Time (sec)",
                    profile["time_options"],
                    key=key_t,
                )
                sets_rows.append({"time_sec": int(t)})

            else:
                c1, c2 = st.columns(2)
                key_w = f"{ns}_w_{idx}"
                key_r = f"{ns}_r_{idx}"

                if key_w not in st.session_state:
                    st.session_state[key_w] = int(s.get("weight", 0) or 0)
                if key_r not in st.session_state:
                    st.session_state[key_r] = int(s.get("reps", 0) or 0)

                with c1:
                    w = st.selectbox(
                        f"Set {idx} — Weight (kg)",
                        profile["weight_options"],
                        key=key_w,
                    )
                with c2:
                    r = st.selectbox(
                        f"Set {idx} — Reps",
                        profile["reps_options"],
                        key=key_r,
                    )

                sets_rows.append({"weight": int(w), "reps": int(r)})

        # Apply sets button INSIDE the box
        apply_btn = st.button("✅ Apply sets", use_container_width=False)

    # Apply action: just sync sets_key with what is currently selected
    if apply_btn:
        st.session_state[sets_key] = sets_rows
        st.rerun()

    # buttons BELOW the box (как на твоём скрине)
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1], gap="small")

    add_btn = c1.button("➕ Add set", use_container_width=True)
    remove_btn = c2.button(
        "➖ Remove set",
        use_container_width=True,
        disabled=len(st.session_state[sets_key]) <= 1,
    )

    if add_btn:
        # фиксируем текущие значения (то, что реально в селектах)
        st.session_state[sets_key] = sets_rows

        if mode == "time":
            last_val = int(sets_rows[-1].get("time_sec", 0))
            st.session_state[sets_key].append({"time_sec": last_val})
            new_idx = len(st.session_state[sets_key])
            st.session_state[f"{ns}_t_{new_idx}"] = last_val
        else:
            last_w = int(sets_rows[-1].get("weight", 0))
            last_r = int(sets_rows[-1].get("reps", 0))
            st.session_state[sets_key].append({"weight": last_w, "reps": last_r})
            new_idx = len(st.session_state[sets_key])
            st.session_state[f"{ns}_w_{new_idx}"] = last_w
            st.session_state[f"{ns}_r_{new_idx}"] = last_r

        st.rerun()

    if remove_btn:
        st.session_state[sets_key] = sets_rows
        if len(st.session_state[sets_key]) > 1:
            last_idx = len(st.session_state[sets_key])
            st.session_state[sets_key] = st.session_state[sets_key][:-1]
            st.session_state.pop(f"{ns}_w_{last_idx}", None)
            st.session_state.pop(f"{ns}_r_{last_idx}", None)
            st.session_state.pop(f"{ns}_t_{last_idx}", None)
        st.rerun()

    # =========================
    # SESSION SUMMARY (uses saved state)
    # =========================
    st.markdown("### Session summary")
    current_sets = st.session_state[sets_key]

    # (не обязательно) показать простым текстом:
    if mode == "time":
        txt = " | ".join([f"{int(s.get('time_sec', 0))}s" for s in current_sets])
    else:
        txt = " | ".join([f"{int(s.get('weight', 0))}×{int(s.get('reps', 0))}" for s in current_sets])
    st.caption(txt)

    # =========================
    # SAVE WORKOUT
    # =========================
    if st.button("💾 Save workout", key=f"{ns}_save_btn"):
        try:
            current_sets = st.session_state[sets_key]

            if mode == "time":
                cleaned = [s for s in current_sets if int(s.get("time_sec", 0)) > 0]
                normalized = [{"weight": 0, "reps": 0, "time_sec": int(s["time_sec"])} for s in cleaned]
            else:
                cleaned = [s for s in current_sets if int(s.get("weight", 0)) > 0 and int(s.get("reps", 0)) > 0]
                normalized = [{"weight": int(s["weight"]), "reps": int(s["reps"]), "time_sec": None} for s in cleaned]

            if not normalized:
                st.error("Add at least one filled set.")
                st.stop()

            ex_id = upsert_exercise(conn, exercise_name)
            insert_workout(conn, str(workout_date), ex_id, normalized)

            st.success("Saved ✅")

            # reset state and widget keys
            st.session_state[sets_key] = [{"time_sec": 0}] if mode == "time" else [{"weight": 0, "reps": 0}]
            for i in range(1, 60):
                st.session_state.pop(f"{ns}_w_{i}", None)
                st.session_state.pop(f"{ns}_r_{i}", None)
                st.session_state.pop(f"{ns}_t_{i}", None)

            st.rerun()

        except Exception as e:
            st.error(f"Save failed: {e}")



# =========================
# TAB: History
# =========================

with tab_history:
    st.subheader("History")

    # --- Filters ---
    ex_df = get_exercises_df()
    ex_list = ["All"] + ex_df["name"].tolist()
    c1, c2, c3 = st.columns([2, 2, 2])

    with c1:
        ex_filter = st.selectbox("Exercise", ex_list, index=0, key="hist_ex_filter")

    with conn.cursor() as cur:
        cur.execute("SELECT MIN(workout_date), MAX(workout_date) FROM workouts")
        row = cur.fetchone()

    if not row or row[0] is None:
        st.info("No workouts yet.")
        st.stop()

    dmin = row[0]
    dmax = row[1]
    default_from = max(dmin, dmax - timedelta(days=30))

    with c2:
        d_from = st.date_input("From", value=default_from, min_value=dmin, max_value=dmax, key="hist_from")
    with c3:
        d_to = st.date_input("To", value=dmax, min_value=dmin, max_value=dmax, key="hist_to")

    t0 = time.time()
    view = get_history_compact(ex_filter, str(d_from), str(d_to))
    dbg(f"history_compact: {time.time() - t0:.3f}s | rows={len(view)}")

    if view.empty:
        st.info("No records for current filters.")
        st.stop()

    st.markdown("---")

    # --- Day groups (custom accordion + copy button) ---
    for day, day_df in view.groupby("workout_date", sort=False):
        day_df = day_df.sort_values("workout_id", ascending=False)

        # Text for copying ONLY this day
        lines = [f"📅 {day} · {len(day_df)} exercises"]
        for _, rr in day_df.iterrows():
            sets = ", ".join([s.strip() for s in str(rr["sets"]).split("|")])
            lines.append(f"{rr['exercise']}: {sets}")
        day_text = "\n".join(lines)

        # Safe keys
        day_id = re.sub(r"[^0-9A-Za-z_]+", "_", str(day))
        open_key = f"open_day_{day_id}"
        toggle_key = f"toggle_day_{day_id}"
        copy_key = f"copy_day_{day_id}"

        if open_key not in st.session_state:
            st.session_state[open_key] = False

        is_open = bool(st.session_state[open_key])
        arrow = "▼" if is_open else "▶"

        # Header row: left is "accordion button", right is copy button
        left_col, right_col = st.columns([10, 2], vertical_alignment="center")

        with left_col:
            if st.button(
                f"{arrow} 📅 {day} · {len(day_df)} entries",
                key=toggle_key,
                use_container_width=True,
            ):
                st.session_state[open_key] = not st.session_state[open_key]
                st.rerun()

        with right_col:
            copy_to_clipboard_button(day_text, key=copy_key, label="📋 Copy")

        # Expanded content
        if st.session_state[open_key]:
            for _, r in day_df.iterrows():
                workout_id = int(r["workout_id"])

                l, m, rr = st.columns([6, 2, 2], vertical_alignment="center")

                with l:
                    st.markdown(f"**{r['exercise']}**")
                    chips = "".join(
                        [f'<span class="set-chip">{s.strip()}</span>' for s in str(r["sets"]).split("|")]
                    )
                    st.markdown(f'<div class="sets-wrap">{chips}</div>', unsafe_allow_html=True)

                with m:
                    pop = st.popover("✏️ Edit", use_container_width=True)
                    with pop:
                        st.info("Edit UI пока не подключён (можно добавить позже).")
                        data = get_workout_for_edit(conn, workout_id)
                        st.write(
                            {
                                "date": data["workout_date"],
                                "exercise": data["exercise"],
                                "sets_rows": len(data["sets"]),
                            }
                        )

                with rr:
                    pop = st.popover("🗑 Delete", use_container_width=True)
                    with pop:
                        st.write("Delete this entry?")
                        confirm = st.checkbox("Confirm", key=f"confirm_{workout_id}")
                        if st.button("Delete", key=f"del_{workout_id}", disabled=not confirm):
                            delete_workout(conn, workout_id)
                            st.success("Deleted ✅")
                            st.rerun()

            st.markdown("---")

# =========================
# TAB: Progress
# =========================
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

    entries_map: dict[int, int] = {}
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

    headers = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
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
                cnt = (
                    f'<span class="cal-count">{entries_map[d]} записи</span>'
                    if trained
                    else '<span class="cal-count">&nbsp;</span>'
                )
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

    # Ensure datetime on x-axis
    df["workout_date"] = pd.to_datetime(df["workout_date"])

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