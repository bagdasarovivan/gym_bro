import streamlit as st
import calendar
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta
import time
import json
import hashlib
import re
import os
from contextlib import contextmanager
import psycopg2
import psycopg2.extras
from psycopg2 import OperationalError, InterfaceError
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

/* Workout card in History */
.workout-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.workout-card-exercise {
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 6px;
}
.pr-badge {
    display: inline-block;
    background: rgba(255, 215, 0, 0.18);
    border: 1px solid rgba(255, 215, 0, 0.4);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 12px;
    margin-left: 8px;
    color: #FFD700;
}

/* Streak badge */
.streak-box {
    background: linear-gradient(135deg, rgba(255,100,0,0.15), rgba(255,60,0,0.08));
    border: 1px solid rgba(255,100,0,0.3);
    border-radius: 14px;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
}
.streak-number {
    font-size: 42px;
    font-weight: 900;
    color: #FF6400;
    line-height: 1;
}
.streak-label {
    font-size: 15px;
    opacity: 0.85;
}

/* PR progress bar */
.pr-progress-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 14px 18px;
    margin: 12px 0;
}
.pr-bar-bg {
    background: rgba(255,255,255,0.1);
    border-radius: 99px;
    height: 10px;
    margin-top: 8px;
    overflow: hidden;
}
.pr-bar-fill {
    height: 10px;
    border-radius: 99px;
    background: linear-gradient(90deg, #00C853, #69F0AE);
    transition: width 0.4s ease;
}

/* Weekly volume bars */
.vol-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}
.vol-label { font-size: 13px; opacity: 0.75; width: 90px; flex-shrink: 0; }
.vol-bar-bg {
    flex: 1;
    background: rgba(255,255,255,0.08);
    border-radius: 99px;
    height: 18px;
    overflow: hidden;
}
.vol-bar-fill {
    height: 18px;
    border-radius: 99px;
    background: linear-gradient(90deg, #2979FF, #82B1FF);
}
.vol-value { font-size: 13px; opacity: 0.85; width: 80px; text-align: right; flex-shrink: 0; }
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
def _new_conn():
    db_url = st.secrets.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("Нет DATABASE_URL в st.secrets (Supabase не подключён)")
    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    return conn


@st.cache_resource
def get_conn():
    return _new_conn()


def get_live_conn():
    conn = get_conn()
    if conn.closed:
        get_conn.clear()
        return get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.commit()
    except (OperationalError, InterfaceError):
        get_conn.clear()
        conn = get_conn()
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
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_sets_workout_setno ON sets(workout_id, set_no);")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS app_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
    conn.commit()


def _seed_hash(names: list[str]) -> str:
    normalized = sorted([n.strip() for n in names if n and n.strip()])
    payload = json.dumps(normalized, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def seed_exercises_hashed(conn, names: list[str]):
    h = _seed_hash(names)
    clean = [(n.strip(),) for n in names if n and n.strip()]
    if clean:
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM app_meta WHERE key = %s", ("seed_hash_exercises",))
            row = cur.fetchone()
            if row and str(row[0]) == h:
                return
            cur.executemany(
                "INSERT INTO exercises(name) VALUES(%s) ON CONFLICT (name) DO NOTHING",
                clean,
            )
            cur.execute(
                """
                INSERT INTO app_meta(key, value)
                VALUES(%s, %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """,
                ("seed_hash_exercises", h),
            )
        conn.commit()
        get_exercises_df.clear()


@st.cache_resource
def ensure_db_ready():
    conn = get_conn()
    init_db(conn)
    seed_exercises_hashed(conn, SEED_EXERCISES)


# =========================
# CACHED READS
# =========================
@st.cache_data(ttl=600)
def get_exercises_df() -> pd.DataFrame:
    conn = get_live_conn()
    return pd.read_sql_query("SELECT id, name FROM exercises ORDER BY name", conn)


@st.cache_data(ttl=600)
def get_month_daily_df(year: int, month: int) -> pd.DataFrame:
    conn = get_live_conn()
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
    conn = get_live_conn()
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
        ORDER BY w.workout_date DESC, w.id ASC
    """
    if exercise and exercise != "All":
        q = base.format(ex_filter="AND e.name = %s")
        return pd.read_sql_query(q, conn, params=(d_from, d_to, exercise))
    q = base.format(ex_filter="")
    return pd.read_sql_query(q, conn, params=(d_from, d_to))


@st.cache_data(ttl=300)
def get_progress_daily(exercise_name: str) -> pd.DataFrame:
    conn = get_live_conn()
    return pd.read_sql_query(
        """
        SELECT
            w.workout_date,
            MAX(s.weight * (1 + (s.reps / 30.0))) AS est_1rm,
            MAX(s.weight) AS top_weight
        FROM workouts w
        JOIN exercises e ON e.id = w.exercise_id
        JOIN sets s ON s.workout_id = w.id
        WHERE e.name = %s
        AND (s.time_sec IS NULL OR s.time_sec = 0)
        AND s.weight > 0 AND s.reps > 0
        GROUP BY w.workout_date
        ORDER BY w.workout_date ASC
        """,
        conn,
        params=(exercise_name,),
    )


@st.cache_data(ttl=300)
def get_history_date_bounds() -> tuple[date | None, date | None]:
    conn = get_live_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT MIN(workout_date), MAX(workout_date) FROM workouts")
        row = cur.fetchone()
    if not row or not row[0]:
        return (None, None)
    return (row[0], row[1])


# =========================
# NEW: STREAK
# =========================
@st.cache_data(ttl=300)
def get_training_streak() -> int:
    """Count consecutive days with at least one workout ending today or yesterday."""
    conn = get_live_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT workout_date FROM workouts ORDER BY workout_date DESC")
        rows = cur.fetchall()
    if not rows:
        return 0

    dates = sorted({r[0] for r in rows}, reverse=True)
    today = date.today()
    # Start from today or yesterday
    if dates[0] < today - timedelta(days=1):
        return 0

    streak = 0
    expected = dates[0]
    for d in dates:
        if d == expected:
            streak += 1
            expected = expected - timedelta(days=1)
        elif d < expected:
            break
    return streak


# =========================
# NEW: PERSONAL RECORDS
# =========================
@st.cache_data(ttl=300)
def get_all_prs() -> dict[str, float]:
    """Return best est_1rm per exercise."""
    conn = get_live_conn()
    df = pd.read_sql_query(
        """
        SELECT e.name AS exercise,
               MAX(s.weight * (1 + s.reps / 30.0)) AS best_1rm
        FROM sets s
        JOIN workouts w ON w.id = s.workout_id
        JOIN exercises e ON e.id = w.exercise_id
        WHERE s.weight > 0 AND s.reps > 0
          AND (s.time_sec IS NULL OR s.time_sec = 0)
        GROUP BY e.name
        """,
        conn,
    )
    if df.empty:
        return {}
    return dict(zip(df["exercise"], df["best_1rm"]))


@st.cache_data(ttl=300)
def get_workout_prs() -> dict[int, bool]:
    """Return set of workout_ids that contain a PR (best 1RM on that date)."""
    conn = get_live_conn()
    df = pd.read_sql_query(
        """
        WITH ranked AS (
            SELECT
                w.id AS workout_id,
                e.name AS exercise,
                MAX(s.weight * (1 + s.reps / 30.0)) AS session_1rm,
                MAX(MAX(s.weight * (1 + s.reps / 30.0)))
                    OVER (PARTITION BY e.name ORDER BY w.workout_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                    AS prev_best
            FROM sets s
            JOIN workouts w ON w.id = s.workout_id
            JOIN exercises e ON e.id = w.exercise_id
            WHERE s.weight > 0 AND s.reps > 0
              AND (s.time_sec IS NULL OR s.time_sec = 0)
            GROUP BY w.id, e.name, w.workout_date
        )
        SELECT workout_id FROM ranked
        WHERE prev_best IS NULL OR session_1rm > prev_best
        """,
        conn,
    )
    if df.empty:
        return {}
    return {int(wid): True for wid in df["workout_id"]}


# =========================
# NEW: WEEKLY VOLUME
# =========================
@st.cache_data(ttl=300)
def get_weekly_volume(exercise_name: str, weeks: int = 8) -> pd.DataFrame:
    """Total volume (weight × reps) per week for an exercise."""
    conn = get_live_conn()
    return pd.read_sql_query(
        """
        SELECT
            DATE_TRUNC('week', w.workout_date) AS week_start,
            SUM(s.weight * s.reps) AS volume
        FROM sets s
        JOIN workouts w ON w.id = s.workout_id
        JOIN exercises e ON e.id = w.exercise_id
        WHERE e.name = %s
          AND s.weight > 0 AND s.reps > 0
          AND (s.time_sec IS NULL OR s.time_sec = 0)
          AND w.workout_date >= CURRENT_DATE - INTERVAL '%s weeks'
        GROUP BY DATE_TRUNC('week', w.workout_date)
        ORDER BY week_start ASC
        """,
        conn,
        params=(exercise_name, weeks),
    )


def clear_cache_after_write():
    get_exercises_df.clear()
    get_month_daily_df.clear()
    get_history_compact.clear()
    get_progress_daily.clear()
    get_history_date_bounds.clear()
    get_workout_for_edit_cached.clear()
    get_training_streak.clear()
    get_all_prs.clear()
    get_workout_prs.clear()
    get_weekly_volume.clear()


# =========================
# WRITES
# =========================
def upsert_exercise(conn, name: str) -> int:
    name = name.strip()
    if not name:
        raise ValueError("Empty exercise name")
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO exercises(name) VALUES(%s)
            ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
            RETURNING id
            """,
            (name,),
        )
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
            payload.append((
                workout_id, i,
                float(s.get("weight", 0)), int(s.get("reps", 0)),
                int(s["time_sec"]) if s.get("time_sec") is not None else None,
            ))
        cur.executemany(
            "INSERT INTO sets(workout_id, set_no, weight, reps, time_sec) VALUES(%s, %s, %s, %s, %s)",
            payload,
        )
    conn.commit()
    clear_cache_after_write()


def delete_workout(conn, workout_id: int):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM workouts WHERE id = %s", (workout_id,))
    conn.commit()
    clear_cache_after_write()


def update_workout(conn, workout_id: int, workout_date: str, exercise_id: int, sets_rows: list[dict]):
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE workouts SET workout_date = %s, exercise_id = %s WHERE id = %s",
            (workout_date, exercise_id, workout_id),
        )
        cur.execute("DELETE FROM sets WHERE workout_id = %s", (workout_id,))
        payload: list[tuple] = []
        for i, s in enumerate(sets_rows, start=1):
            payload.append((
                workout_id, i,
                float(s.get("weight", 0)), int(s.get("reps", 0)),
                int(s["time_sec"]) if s.get("time_sec") is not None else None,
            ))
        cur.executemany(
            "INSERT INTO sets(workout_id, set_no, weight, reps, time_sec) VALUES(%s, %s, %s, %s, %s)",
            payload,
        )
    conn.commit()
    clear_cache_after_write()


def get_workout_for_edit(conn, workout_id: int) -> dict:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                w.id AS workout_id,
                w.workout_date,
                e.name AS exercise,
                COALESCE(
                    json_agg(
                        json_build_object(
                            'set_no', s.set_no,
                            'weight', s.weight,
                            'reps', s.reps,
                            'time_sec', s.time_sec
                        )
                        ORDER BY s.set_no
                    ) FILTER (WHERE s.id IS NOT NULL),
                    '[]'::json
                ) AS sets
            FROM workouts w
            JOIN exercises e ON e.id = w.exercise_id
            LEFT JOIN sets s ON s.workout_id = w.id
            WHERE w.id = %s
            GROUP BY w.id, w.workout_date, e.name
            """,
            (workout_id,),
        )
        row = cur.fetchone()
    if not row:
        raise ValueError("Workout not found")
    sets_df = pd.DataFrame(row["sets"])
    return {
        "workout_id": int(row["workout_id"]),
        "workout_date": str(row["workout_date"]),
        "exercise": str(row["exercise"]),
        "sets": sets_df,
    }


@st.cache_data(ttl=300)
def get_workout_for_edit_cached(workout_id: int) -> dict:
    conn = get_live_conn()
    return get_workout_for_edit(conn, workout_id)


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
# HELPERS
# =========================
def plural_ru(n: int, one: str, few: str, many: str) -> str:
    """Russian plural: 1 тренировка, 2 тренировки, 5 тренировок"""
    if 11 <= (n % 100) <= 19:
        return many
    r = n % 10
    if r == 1:
        return one
    if 2 <= r <= 4:
        return few
    return many


def plural_entries(n: int) -> str:
    return f"{n} {plural_ru(n, 'запись', 'записи', 'записей')}"


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
    conn = get_live_conn()
    ensure_db_ready()
except Exception as e:
    st.error(f"DB connect failed: {e}")
    st.stop()

tab_add, tab_history, tab_progress = st.tabs(["➕ Add workout", "📜 History", "📈 Progress"])

# =========================
# TAB: Add workout
# =========================
with tab_add:
    st.subheader("Add workout")

    # --- Streak banner ---
    streak = get_training_streak()
    if streak >= 2:
        st.markdown(
            f"""
            <div class="streak-box">
                <div class="streak-number">{streak}🔥</div>
                <div>
                    <div style="font-weight:700; font-size:16px;">Streak!</div>
                    <div class="streak-label">{streak} {plural_ru(streak, 'день подряд', 'дня подряд', 'дней подряд')} — не останавливайся!</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif streak == 1:
        st.markdown(
            """
            <div class="streak-box">
                <div class="streak-number">1🔥</div>
                <div>
                    <div style="font-weight:700; font-size:16px;">Хорошее начало!</div>
                    <div class="streak-label">Приходи завтра и серия продолжится</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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

    if "add_ns_by_ex" not in st.session_state:
        st.session_state.add_ns_by_ex = {}
    if exercise_name not in st.session_state.add_ns_by_ex:
        st.session_state.add_ns_by_ex[exercise_name] = uuid.uuid4().hex[:10]

    ns = f"add_{st.session_state.add_ns_by_ex[exercise_name]}"
    sets_key = f"sets_{ns}"

    if sets_key not in st.session_state:
        st.session_state[sets_key] = [{"time_sec": 0}] if mode == "time" else [{"weight": 0, "reps": 0}]

    with st.container():
        for idx, s in enumerate(st.session_state[sets_key], start=1):
            if mode == "time":
                key_t = f"{ns}_t_{idx}"
                if key_t not in st.session_state:
                    st.session_state[key_t] = int(s.get("time_sec", 0) or 0)
                st.selectbox(f"Set {idx} — Time (sec)", profile["time_options"], key=key_t)
            else:
                c1, c2 = st.columns(2)
                key_w = f"{ns}_w_{idx}"
                key_r = f"{ns}_r_{idx}"
                if key_w not in st.session_state:
                    st.session_state[key_w] = int(s.get("weight", 0) or 0)
                if key_r not in st.session_state:
                    st.session_state[key_r] = int(s.get("reps", 0) or 0)
                with c1:
                    st.selectbox(f"Set {idx} — Weight (kg)", profile["weight_options"], key=key_w)
                with c2:
                    st.selectbox(f"Set {idx} — Reps", profile["reps_options"], key=key_r)

        apply_btn = st.button("✅ Apply sets", use_container_width=False)

    st.session_state[sets_key] = read_sets_from_widgets(ns, len(st.session_state[sets_key]), mode)

    if apply_btn:
        st.session_state[sets_key] = read_sets_from_widgets(ns, len(st.session_state[sets_key]), mode)

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1], gap="small")

    add_btn = c1.button("➕ Add set", use_container_width=True)
    remove_btn = c2.button(
        "➖ Remove set",
        use_container_width=True,
        disabled=len(st.session_state[sets_key]) <= 1,
    )

    if add_btn:
        current_rows = read_sets_from_widgets(ns, len(st.session_state[sets_key]), mode)
        st.session_state[sets_key] = current_rows
        if mode == "time":
            last_val = int(current_rows[-1].get("time_sec", 0))
            st.session_state[sets_key].append({"time_sec": last_val})
            st.session_state[f"{ns}_t_{len(st.session_state[sets_key])}"] = last_val
        else:
            last_w = int(current_rows[-1].get("weight", 0))
            last_r = int(current_rows[-1].get("reps", 0))
            st.session_state[sets_key].append({"weight": last_w, "reps": last_r})
            st.session_state[f"{ns}_w_{len(st.session_state[sets_key])}"] = last_w
            st.session_state[f"{ns}_r_{len(st.session_state[sets_key])}"] = last_r
        st.rerun()

    if remove_btn:
        current_rows = read_sets_from_widgets(ns, len(st.session_state[sets_key]), mode)
        st.session_state[sets_key] = current_rows
        if len(st.session_state[sets_key]) > 1:
            last_idx = len(st.session_state[sets_key])
            st.session_state[sets_key] = st.session_state[sets_key][:-1]
            st.session_state.pop(f"{ns}_w_{last_idx}", None)
            st.session_state.pop(f"{ns}_r_{last_idx}", None)
            st.session_state.pop(f"{ns}_t_{last_idx}", None)
        st.rerun()

    st.markdown("### Session summary")
    current_sets = st.session_state[sets_key]
    if mode == "time":
        txt = " | ".join([f"{int(s.get('time_sec', 0))}s" for s in current_sets])
    else:
        txt = " | ".join([f"{int(s.get('weight', 0))}×{int(s.get('reps', 0))}" for s in current_sets])
    st.caption(txt)

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
            else:
                ex_id = upsert_exercise(conn, exercise_name)
                insert_workout(conn, str(workout_date), ex_id, normalized)
                st.success("Saved ✅")
                st.session_state[sets_key] = [{"time_sec": 0}] if mode == "time" else [{"weight": 0, "reps": 0}]
                for i in range(1, 60):
                    st.session_state.pop(f"{ns}_w_{i}", None)
                    st.session_state.pop(f"{ns}_r_{i}", None)
                    st.session_state.pop(f"{ns}_t_{i}", None)
                st.rerun()
        except Exception as e:
            conn.rollback()
            st.error(f"Save failed: {e}")


# =========================
# TAB: History
# =========================
with tab_history:
    st.subheader("History")

    ex_df = get_exercises_df()
    ex_list = ["All"] + ex_df["name"].tolist()
    c1, c2, c3 = st.columns([2, 2, 2])

    with c1:
        ex_filter = st.selectbox("Exercise", ex_list, index=0, key="hist_ex_filter")

    dmin, dmax = get_history_date_bounds()
    if dmin is None:
        st.info("No workouts yet.")
    else:
        default_from = max(dmin, dmax - timedelta(days=30))

        with c2:
            d_from = st.date_input("From", value=default_from, min_value=dmin, max_value=dmax, key="hist_from")
        with c3:
            d_to = st.date_input("To", value=dmax, min_value=dmin, max_value=dmax, key="hist_to")

        if d_from > d_to:
            st.warning("'From' не может быть больше 'To'.")
        else:
            t0 = time.time()
            view = get_history_compact(ex_filter, str(d_from), str(d_to))
            dbg(f"history_compact: {time.time() - t0:.3f}s | rows={len(view)}")

            if view.empty:
                st.info("No records for current filters.")
            else:
                # Load PR data for badges
                pr_workout_ids = get_workout_prs()

                st.markdown("---")

                for day, day_df in view.groupby("workout_date", sort=False):
                    lines = [f"📅 {day} · {len(day_df)} exercises"]
                    for _, rr in day_df.iterrows():
                        sets = ", ".join([s.strip() for s in str(rr["sets"]).split("|")])
                        lines.append(f"{rr['exercise']}: {sets}")
                    day_text = "\n".join(lines)

                    day_id = re.sub(r"[^0-9A-Za-z_]+", "_", str(day))
                    open_key = f"open_day_{day_id}"
                    toggle_key = f"toggle_day_{day_id}"
                    copy_key = f"copy_day_{day_id}"

                    if open_key not in st.session_state:
                        st.session_state[open_key] = False

                    is_open = bool(st.session_state[open_key])
                    arrow = "▼" if is_open else "▶"

                    left_col, right_col = st.columns([10, 2], vertical_alignment="center")
                    with left_col:
                        # Fixed: correct plural form for entries count
                        label = plural_entries(len(day_df))
                        if st.button(
                            f"{arrow} 📅 {day} · {label}",
                            key=toggle_key,
                            use_container_width=True,
                        ):
                            st.session_state[open_key] = not st.session_state[open_key]
                            st.rerun()
                    with right_col:
                        copy_to_clipboard_button(day_text, key=copy_key, label="📋 Copy")

                    if st.session_state[open_key]:
                        for _, r in day_df.iterrows():
                            workout_id = int(r["workout_id"])
                            is_pr = pr_workout_ids.get(workout_id, False)

                            # Workout card
                            pr_badge = '<span class="pr-badge">🏆 PR</span>' if is_pr else ''
                            chips = "".join(
                                [f'<span class="set-chip">{s.strip()}</span>' for s in str(r["sets"]).split("|")]
                            )
                            st.markdown(
                                f"""
                                <div class="workout-card">
                                    <div class="workout-card-exercise">{r['exercise']}{pr_badge}</div>
                                    <div class="sets-wrap">{chips}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            action_l, action_m, action_r = st.columns([6, 2, 2])

                            with action_m:
                                pop = st.popover("✏️ Edit", use_container_width=True)
                                with pop:
                                    data = get_workout_for_edit_cached(workout_id)
                                    edit_ns = f"edit_{workout_id}"
                                    edit_sets_key = f"{edit_ns}_sets"
                                    edit_type_key = f"{edit_ns}_type"

                                    if edit_sets_key not in st.session_state:
                                        initial_sets: list[dict] = []
                                        for _, srow in data["sets"].iterrows():
                                            t = int(srow.get("time_sec", 0) or 0)
                                            if t > 0:
                                                initial_sets.append({"time_sec": t})
                                            else:
                                                initial_sets.append({
                                                    "weight": int(float(srow.get("weight", 0) or 0)),
                                                    "reps": int(float(srow.get("reps", 0) or 0)),
                                                })
                                        if not initial_sets:
                                            initial_sets = [{"weight": 0, "reps": 0}]
                                        st.session_state[edit_sets_key] = initial_sets

                                    if edit_type_key not in st.session_state:
                                        st.session_state[edit_type_key] = EXERCISE_TYPE.get(data["exercise"], "light")

                                    edit_date_key = f"{edit_ns}_date"
                                    if edit_date_key not in st.session_state:
                                        st.session_state[edit_date_key] = pd.to_datetime(data["workout_date"]).date()

                                    edit_ex_key = f"{edit_ns}_exercise"
                                    if edit_ex_key not in st.session_state:
                                        st.session_state[edit_ex_key] = data["exercise"]

                                    edit_exercise = st.selectbox("Exercise", ex_df["name"].tolist(), key=edit_ex_key)
                                    st.date_input(
                                        "📅 Date",
                                        min_value=date(2020, 1, 1),
                                        max_value=date.today(),
                                        key=edit_date_key,
                                    )

                                    new_type = EXERCISE_TYPE.get(edit_exercise, "light")
                                    prev_type = st.session_state[edit_type_key]
                                    if new_type != prev_type:
                                        st.session_state[edit_type_key] = new_type
                                        st.session_state[edit_sets_key] = (
                                            [{"time_sec": 0}] if new_type == "timed" else [{"weight": 0, "reps": 0}]
                                        )
                                        for i in range(1, 60):
                                            st.session_state.pop(f"{edit_ns}_w_{i}", None)
                                            st.session_state.pop(f"{edit_ns}_r_{i}", None)
                                            st.session_state.pop(f"{edit_ns}_t_{i}", None)
                                        st.rerun()

                                    edit_profile = TYPE_PROFILES[new_type]
                                    edit_mode = edit_profile["mode"]

                                    for idx, s in enumerate(st.session_state[edit_sets_key], start=1):
                                        if edit_mode == "time":
                                            key_t = f"{edit_ns}_t_{idx}"
                                            if key_t not in st.session_state:
                                                st.session_state[key_t] = int(s.get("time_sec", 0) or 0)
                                            st.selectbox(f"Set {idx} — Time (sec)", edit_profile["time_options"], key=key_t)
                                        else:
                                            cew, cer = st.columns(2)
                                            key_w = f"{edit_ns}_w_{idx}"
                                            key_r = f"{edit_ns}_r_{idx}"
                                            if key_w not in st.session_state:
                                                st.session_state[key_w] = int(s.get("weight", 0) or 0)
                                            if key_r not in st.session_state:
                                                st.session_state[key_r] = int(s.get("reps", 0) or 0)
                                            with cew:
                                                st.selectbox(f"Set {idx} — Weight (kg)", edit_profile["weight_options"], key=key_w)
                                            with cer:
                                                st.selectbox(f"Set {idx} — Reps", edit_profile["reps_options"], key=key_r)

                                    st.session_state[edit_sets_key] = read_sets_from_widgets(
                                        edit_ns, len(st.session_state[edit_sets_key]), edit_mode
                                    )

                                    e1, e2 = st.columns(2)
                                    if e1.button("➕ Add set", key=f"{edit_ns}_add_set", use_container_width=True):
                                        current_rows = read_sets_from_widgets(edit_ns, len(st.session_state[edit_sets_key]), edit_mode)
                                        st.session_state[edit_sets_key] = current_rows
                                        if edit_mode == "time":
                                            v = int(current_rows[-1].get("time_sec", 0))
                                            st.session_state[edit_sets_key].append({"time_sec": v})
                                            st.session_state[f"{edit_ns}_t_{len(st.session_state[edit_sets_key])}"] = v
                                        else:
                                            w = int(current_rows[-1].get("weight", 0))
                                            reps = int(current_rows[-1].get("reps", 0))
                                            st.session_state[edit_sets_key].append({"weight": w, "reps": reps})
                                            st.session_state[f"{edit_ns}_w_{len(st.session_state[edit_sets_key])}"] = w
                                            st.session_state[f"{edit_ns}_r_{len(st.session_state[edit_sets_key])}"] = reps
                                        st.rerun()

                                    if e2.button("➖ Remove set", key=f"{edit_ns}_rm_set", use_container_width=True,
                                                 disabled=len(st.session_state[edit_sets_key]) <= 1):
                                        current_rows = read_sets_from_widgets(edit_ns, len(st.session_state[edit_sets_key]), edit_mode)
                                        st.session_state[edit_sets_key] = current_rows
                                        last_idx = len(st.session_state[edit_sets_key])
                                        st.session_state[edit_sets_key] = st.session_state[edit_sets_key][:-1]
                                        st.session_state.pop(f"{edit_ns}_w_{last_idx}", None)
                                        st.session_state.pop(f"{edit_ns}_r_{last_idx}", None)
                                        st.session_state.pop(f"{edit_ns}_t_{last_idx}", None)
                                        st.rerun()

                                    s1, s2 = st.columns(2)
                                    if s1.button("Save changes", key=f"{edit_ns}_save", use_container_width=True):
                                        try:
                                            rows = st.session_state[edit_sets_key]
                                            if edit_mode == "time":
                                                cleaned = [x for x in rows if int(x.get("time_sec", 0)) > 0]
                                                normalized = [{"weight": 0, "reps": 0, "time_sec": int(x["time_sec"])} for x in cleaned]
                                            else:
                                                cleaned = [x for x in rows if int(x.get("weight", 0)) > 0 and int(x.get("reps", 0)) > 0]
                                                normalized = [{"weight": int(x["weight"]), "reps": int(x["reps"]), "time_sec": None} for x in cleaned]
                                            if not normalized:
                                                st.error("Add at least one filled set.")
                                            else:
                                                ex_id = int(ex_df.loc[ex_df["name"] == edit_exercise, "id"].iloc[0])
                                                update_workout(conn, workout_id, str(st.session_state[edit_date_key]), ex_id, normalized)
                                                st.success("Updated ✅")
                                                st.rerun()
                                        except Exception as e:
                                            conn.rollback()
                                            st.error(f"Update failed: {e}")

                                    if s2.button("Cancel", key=f"{edit_ns}_cancel", use_container_width=True):
                                        st.rerun()

                            with action_r:
                                pop = st.popover("🗑 Delete", use_container_width=True)
                                with pop:
                                    st.write("Delete this entry?")
                                    confirm = st.checkbox("Confirm", key=f"confirm_{workout_id}")
                                    if st.button("Delete", key=f"del_{workout_id}", disabled=not confirm):
                                        try:
                                            delete_workout(conn, workout_id)
                                            st.success("Deleted ✅")
                                            st.rerun()
                                        except Exception as e:
                                            conn.rollback()
                                            st.error(f"Delete failed: {e}")

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
                # Fixed: correct Russian plural
                cnt = (
                    f'<span class="cal-count">{plural_entries(entries_map[d])}</span>'
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
    else:
        ex = st.selectbox("Exercise", ex_names, key="progress_exercise_select")

        t_p = time.time()
        df = get_progress_daily(ex)
        dbg(f"progress_sets: {time.time() - t_p:.3f}s | rows={len(df)}")

        if df.empty:
            st.info("No weight+reps sets for this exercise yet.")
        else:
            df["workout_date"] = pd.to_datetime(df["workout_date"])

            # --- Dark theme chart ---
            plt.style.use("dark_background")
            fig, ax1 = plt.subplots(figsize=(9, 4))
            fig.patch.set_alpha(0.0)
            ax1.set_facecolor("none")

            color_1rm = "#69F0AE"
            color_tw = "#82B1FF"

            ax1.plot(df["workout_date"], df["est_1rm"], color=color_1rm, linewidth=2.5,
                     marker="o", markersize=5, label="Est. 1RM")
            ax1.set_xlabel("Date", color="white", fontsize=11)
            ax1.set_ylabel("Estimated 1RM (kg)", color=color_1rm, fontsize=11)
            ax1.tick_params(axis="y", labelcolor=color_1rm)
            ax1.tick_params(axis="x", colors="white", rotation=30)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
            ax1.xaxis.set_major_locator(mdates.AutoDateLocator())

            ax2 = ax1.twinx()
            ax2.set_facecolor("none")
            ax2.plot(df["workout_date"], df["top_weight"], color=color_tw, linewidth=2,
                     linestyle="--", marker="s", markersize=4, label="Top weight")
            ax2.set_ylabel("Top weight (kg)", color=color_tw, fontsize=11)
            ax2.tick_params(axis="y", labelcolor=color_tw)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
                       facecolor="#1a1a1a", edgecolor="#444444", labelcolor="white")

            ax1.set_title(f"{ex} — 1RM & Top weight", color="white", fontsize=13, pad=12)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            best_1rm = float(df["est_1rm"].max())
            st.metric("🏆 Best estimated 1RM", f"{best_1rm:.1f} kg")

            # =========================
            # NEW: Progress to next PR
            # =========================
            all_prs = get_all_prs()
            current_pr = all_prs.get(ex, 0)
            last_session_1rm = float(df["est_1rm"].iloc[-1])

            next_pr_target = current_pr * 1.025  # +2.5% above current PR
            progress_pct = min((last_session_1rm / next_pr_target) * 100, 100) if next_pr_target > 0 else 0
            kg_to_pr = max(next_pr_target - last_session_1rm, 0)

            st.markdown(
                f"""
                <div class="pr-progress-wrap">
                    <div style="font-size:15px; font-weight:600;">📊 Progress to next PR</div>
                    <div style="font-size:13px; opacity:0.75; margin-top:4px;">
                        Last session: <b>{last_session_1rm:.1f} kg</b> &nbsp;·&nbsp;
                        Current PR: <b>{current_pr:.1f} kg</b> &nbsp;·&nbsp;
                        {"🏆 PR!" if kg_to_pr == 0 else f"До рекорда: <b>{kg_to_pr:.1f} kg</b>"}
                    </div>
                    <div class="pr-bar-bg">
                        <div class="pr-bar-fill" style="width:{progress_pct:.1f}%"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # =========================
            # NEW: Weekly volume
            # =========================
            st.divider()
            st.markdown("## 📦 Weekly volume")

            vol_df = get_weekly_volume(ex, weeks=10)
            if vol_df.empty:
                st.info("Not enough data for weekly volume.")
            else:
                max_vol = float(vol_df["volume"].max())
                vol_html = ['<div style="margin-top:8px">']
                for _, row in vol_df.iterrows():
                    week_label = pd.to_datetime(row["week_start"]).strftime("%d %b")
                    vol = float(row["volume"])
                    bar_pct = (vol / max_vol * 100) if max_vol > 0 else 0
                    vol_html.append(f"""
                    <div class="vol-row">
                        <div class="vol-label">{week_label}</div>
                        <div class="vol-bar-bg">
                            <div class="vol-bar-fill" style="width:{bar_pct:.1f}%"></div>
                        </div>
                        <div class="vol-value">{int(vol):,} kg</div>
                    </div>
                    """)
                vol_html.append("</div>")
                st.markdown("".join(vol_html), unsafe_allow_html=True)

dbg(f"Render: {time.time() - start_total:.3f} sec")
