import streamlit as st
import calendar
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import time

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy import event

# ---------- DEBUG MODE ----------
DEBUG = False
try:
    DEBUG = bool(st.secrets.get("DEBUG", False))
except Exception:
    DEBUG = False

def dbg(msg: str):
    if DEBUG:
        st.caption(msg)
# --------------------------------

start_total = time.time()

DB_PATH = "fitness.db"

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
# DB / Engine helpers
# =========================
@st.cache_resource
def get_engine() -> Engine:
    """
    If st.secrets["DB_URL"] exists -> PostgreSQL(Supabase).
    Else -> local SQLite.
    """
    if "DB_URL" in st.secrets and str(st.secrets["DB_URL"]).strip():
        db_url = str(st.secrets["DB_URL"]).strip()
        engine = create_engine(db_url, pool_pre_ping=True, future=True)
        return engine

    # SQLite fallback
    sqlite_url = f"sqlite+pysqlite:///{DB_PATH}"
    engine = create_engine(
        sqlite_url,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
        future=True
    )

    # apply PRAGMA on each connection for SQLite
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record):
        try:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("PRAGMA foreign_keys=ON;")
            cursor.close()
        except Exception:
            pass

    return engine


def init_db(engine: Engine):
    ddl_exercises = """
    CREATE TABLE IF NOT EXISTS exercises (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE
    )
    """
    ddl_workouts = """
    CREATE TABLE IF NOT EXISTS workouts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workout_date TEXT NOT NULL,
        exercise_id INTEGER NOT NULL,
        FOREIGN KEY (exercise_id) REFERENCES exercises(id)
    )
    """
    ddl_sets = """
    CREATE TABLE IF NOT EXISTS sets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workout_id INTEGER NOT NULL,
        set_no INTEGER NOT NULL,
        weight REAL NOT NULL,
        reps INTEGER NOT NULL,
        time_sec INTEGER,
        FOREIGN KEY (workout_id) REFERENCES workouts(id)
    )
    """

    # Postgres doesn't have AUTOINCREMENT keyword.
    # We'll swap to SERIAL-like behavior only when needed.
    if engine.dialect.name == "postgresql":
        ddl_exercises = """
        CREATE TABLE IF NOT EXISTS exercises (
            id BIGSERIAL PRIMARY KEY,
            name TEXT NOT NULL UNIQUE
        )
        """
        ddl_workouts = """
        CREATE TABLE IF NOT EXISTS workouts (
            id BIGSERIAL PRIMARY KEY,
            workout_date DATE NOT NULL,
            exercise_id BIGINT NOT NULL REFERENCES exercises(id)
        )
        """
        ddl_sets = """
        CREATE TABLE IF NOT EXISTS sets (
            id BIGSERIAL PRIMARY KEY,
            workout_id BIGINT NOT NULL REFERENCES workouts(id),
            set_no INTEGER NOT NULL,
            weight DOUBLE PRECISION NOT NULL,
            reps INTEGER NOT NULL,
            time_sec INTEGER
        )
        """

    with engine.begin() as c:
        c.execute(text(ddl_exercises))
        c.execute(text(ddl_workouts))
        c.execute(text(ddl_sets))

        # indexes
        if engine.dialect.name == "postgresql":
            c.execute(text("CREATE INDEX IF NOT EXISTS idx_workouts_date ON workouts(workout_date)"))
            c.execute(text("CREATE INDEX IF NOT EXISTS idx_workouts_exercise ON workouts(exercise_id)"))
            c.execute(text("CREATE INDEX IF NOT EXISTS idx_sets_workout ON sets(workout_id)"))
        else:
            c.execute(text("CREATE INDEX IF NOT EXISTS idx_workouts_date ON workouts(workout_date)"))
            c.execute(text("CREATE INDEX IF NOT EXISTS idx_workouts_exercise ON workouts(exercise_id)"))
            c.execute(text("CREATE INDEX IF NOT EXISTS idx_sets_workout ON sets(workout_id)"))


def seed_exercises_once(engine: Engine, names: list[str]):
    """
    Seeds exercises once (idempotent), and does it in ONE batch.
    No per-item commits. No rerun spam.
    """
    # Quick exit if already seeded at least once.
    # If user deletes some rows manually, we still add missing rows below.
    if not names:
        return

    names = [n.strip() for n in names if n and n.strip()]
    if not names:
        return

    if engine.dialect.name == "postgresql":
        sql = text("INSERT INTO exercises(name) VALUES (:name) ON CONFLICT (name) DO NOTHING")
    else:
        sql = text("INSERT OR IGNORE INTO exercises(name) VALUES (:name)")

    payload = [{"name": n} for n in names]

    with engine.begin() as c:
        c.execute(sql, payload)

    # clear cached exercise list because seed could add missing
    get_exercises_df.clear()


# =========================
# Cached reads
# =========================
@st.cache_data(ttl=30)
def get_exercises_df() -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql_query("SELECT id, name FROM exercises ORDER BY name", engine)

@st.cache_data(ttl=30)
def get_history_df() -> pd.DataFrame:
    engine = get_engine()

    if engine.dialect.name == "postgresql":
        q = """
        SELECT
            w.id AS workout_id,
            w.workout_date::text AS workout_date,
            e.name AS exercise,
            s.set_no,
            s.weight,
            s.reps,
            s.time_sec
        FROM workouts w
        JOIN exercises e ON e.id = w.exercise_id
        JOIN sets s ON s.workout_id = w.id
        ORDER BY w.workout_date DESC, w.id DESC, s.set_no ASC
        """
    else:
        q = """
        SELECT
            w.id AS workout_id,
            w.workout_date,
            e.name AS exercise,
            s.set_no,
            s.weight,
            s.reps,
            s.time_sec
        FROM workouts w
        JOIN exercises e ON e.id = w.exercise_id
        JOIN sets s ON s.workout_id = w.id
        ORDER BY w.workout_date DESC, w.id DESC, s.set_no ASC
        """

    return pd.read_sql_query(q, engine)

@st.cache_data(ttl=30)
def get_last_workout_df(exercise_name: str) -> pd.DataFrame | None:
    engine = get_engine()

    if engine.dialect.name == "postgresql":
        q = """
        SELECT
            w.id as workout_id,
            w.workout_date::text AS workout_date,
            s.weight,
            s.reps,
            s.time_sec,
            s.set_no
        FROM workouts w
        JOIN exercises e ON e.id = w.exercise_id
        JOIN sets s ON s.workout_id = w.id
        WHERE e.name = :exercise_name
        ORDER BY w.workout_date DESC, w.id DESC, s.set_no ASC
        """
    else:
        q = """
        SELECT
            w.id as workout_id,
            w.workout_date,
            s.weight,
            s.reps,
            s.time_sec,
            s.set_no
        FROM workouts w
        JOIN exercises e ON e.id = w.exercise_id
        JOIN sets s ON s.workout_id = w.id
        WHERE e.name = :exercise_name
        ORDER BY w.workout_date DESC, w.id DESC, s.set_no ASC
        """

    df = pd.read_sql_query(text(q), engine, params={"exercise_name": exercise_name})
    if df.empty:
        return None

    last_workout_id = int(df.iloc[0]["workout_id"])
    return df[df["workout_id"] == last_workout_id].copy()

@st.cache_data(ttl=60)
def get_month_daily_df(year: int, month: int) -> pd.DataFrame:
    engine = get_engine()
    ym_start = f"{year:04d}-{month:02d}-01"
    last_day = calendar.monthrange(year, month)[1]
    ym_end = f"{year:04d}-{month:02d}-{last_day:02d}"

    if engine.dialect.name == "postgresql":
        q = """
        SELECT workout_date::text AS workout_date, COUNT(*) AS entries
        FROM workouts
        WHERE workout_date BETWEEN :ym_start::date AND :ym_end::date
        GROUP BY workout_date
        """
    else:
        q = """
        SELECT workout_date, COUNT(*) AS entries
        FROM workouts
        WHERE workout_date BETWEEN :ym_start AND :ym_end
        GROUP BY workout_date
        """

    return pd.read_sql_query(text(q), engine, params={"ym_start": ym_start, "ym_end": ym_end})

def clear_all_caches():
    get_exercises_df.clear()
    get_history_df.clear()
    get_last_workout_df.clear()
    get_month_daily_df.clear()

# =========================
# Writes
# =========================
def add_exercise(engine: Engine, name: str) -> int:
    name = name.strip()
    if not name:
        raise ValueError("Empty exercise name")

    if engine.dialect.name == "postgresql":
        ins = text("INSERT INTO exercises(name) VALUES (:name) ON CONFLICT (name) DO NOTHING")
        sel = text("SELECT id FROM exercises WHERE name = :name")
    else:
        ins = text("INSERT OR IGNORE INTO exercises(name) VALUES (:name)")
        sel = text("SELECT id FROM exercises WHERE name = :name")

    with engine.begin() as c:
        c.execute(ins, {"name": name})
        row = c.execute(sel, {"name": name}).fetchone()

    get_exercises_df.clear()
    return int(row[0])

def add_workout_with_sets(engine: Engine, workout_date_str: str, exercise_id: int, sets_rows: list[dict]):
    if not sets_rows:
        raise ValueError("No sets to insert")

    if engine.dialect.name == "postgresql":
        # workout_date is DATE in PG
        ins_w = text("INSERT INTO workouts(workout_date, exercise_id) VALUES (:workout_date::date, :exercise_id) RETURNING id")
    else:
        ins_w = text("INSERT INTO workouts(workout_date, exercise_id) VALUES (:workout_date, :exercise_id)")

    ins_s = text("""
        INSERT INTO sets(workout_id, set_no, weight, reps, time_sec)
        VALUES(:workout_id, :set_no, :weight, :reps, :time_sec)
    """)

    with engine.begin() as c:
        if engine.dialect.name == "postgresql":
            workout_id = int(c.execute(ins_w, {"workout_date": workout_date_str, "exercise_id": exercise_id}).scalar_one())
        else:
            res = c.execute(ins_w, {"workout_date": workout_date_str, "exercise_id": exercise_id})
            # SQLAlchemy gives lastrowid for sqlite inserts
            workout_id = int(res.lastrowid)

        payload = []
        for i, s in enumerate(sets_rows, start=1):
            payload.append({
                "workout_id": workout_id,
                "set_no": i,
                "weight": float(s.get("weight", 0)),
                "reps": int(s.get("reps", 0)),
                "time_sec": int(s["time_sec"]) if s.get("time_sec") is not None else None
            })

        c.execute(ins_s, payload)

    get_history_df.clear()
    get_last_workout_df.clear()
    get_month_daily_df.clear()

def delete_workout(engine: Engine, workout_id: int):
    with engine.begin() as c:
        c.execute(text("DELETE FROM sets WHERE workout_id = :wid"), {"wid": workout_id})
        c.execute(text("DELETE FROM workouts WHERE id = :wid"), {"wid": workout_id})

    get_history_df.clear()
    get_last_workout_df.clear()
    get_month_daily_df.clear()

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Gym BRO",
    page_icon="images/gymbro_icon.png",
    layout="centered"
)

# =========================
# Init DB + seed ONCE
# =========================
engine = get_engine()
init_db(engine)
seed_exercises_once(engine, SEED_EXERCISES)

# =========================
# Styles
# =========================
st.markdown("""
<style>
@media (max-width: 768px) {
    h1 { font-size: 32px !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 40px !important; }
    .stTabs [data-baseweb="tab"] { font-size: 16px !important; padding: 12px 0px !important; }
    img { max-width: 100% !important; height: auto !important; }
    .block-container { padding: 1rem 1rem 2rem 1rem !important; }
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
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 6])
with col1:
    st.image("images/gymbro_logo.png", width=90)
with col2:
    st.markdown("<h1 style='margin:0'>Gym BRO</h1>", unsafe_allow_html=True)

tab_add, tab_history, tab_progress = st.tabs(["➕ Add workout", "📜 History", "📈 Progress"])

# ============================
# TAB: Add workout
# ============================
with tab_add:
    st.subheader("Add workout")

    ex_df = get_exercises_df()
    ex_names = ex_df["name"].tolist()
    if not ex_names:
        st.warning("No exercises found.")
        st.stop()

    date_col, _ = st.columns([1, 4])
    with date_col:
        workout_date = st.date_input(
            "📅 Date",
            value=date.today(),
            min_value=date(2020, 1, 1),
            max_value=date.today(),
            key="workout_date"
        )

    q = st.text_input("Search exercise", "")

    filtered = [x for x in ex_names if q.lower() in x.lower()] if q else ex_names
    fav = [x for x in FAVORITE_EXERCISES if x in filtered]
    rest = [x for x in filtered if x not in fav]
    ex_options = fav + rest

    cA, cB = st.columns([3, 1])
    with cA:
        exercise_name = st.selectbox(
            "Exercise",
            ex_options,
            index=None,
            placeholder="Select exercise",
            key="add_exercise_select"
        )

    if not exercise_name:
        st.stop()

    ns = f"{workout_date}_{exercise_name}".replace(" ", "_")
    sets_key = f"sets_{ns}"

    with cB:
        img_path = EXERCISE_IMAGES.get(exercise_name)
        if img_path:
            st.image(img_path, width=120)

    last_df = get_last_workout_df(exercise_name)
    if last_df is not None:
        st.markdown("#### Last workout")
        date_str = str(last_df.iloc[0]["workout_date"])
        sets_str = []
        for _, row in last_df.iterrows():
            if pd.notna(row["time_sec"]) and int(row["time_sec"]) > 0:
                sets_str.append(f"{int(row['time_sec'])}s")
            else:
                sets_str.append(f"{int(row['weight'])}×{int(row['reps'])}")
        st.caption(f"{date_str} — " + " | ".join(sets_str))

    ex_type = EXERCISE_TYPE.get(exercise_name, "light")
    profile = TYPE_PROFILES[ex_type]

    st.markdown("### Sets")

    # Initialize sets list per namespace
    if sets_key not in st.session_state:
        st.session_state[sets_key] = [{"time_sec": 0}] if profile["mode"] == "time" else [{"weight": 0, "reps": 0}]

    # Ensure mode consistency (if user switches exercise type)
    if profile["mode"] == "time":
        if not st.session_state[sets_key] or "time_sec" not in st.session_state[sets_key][0]:
            st.session_state[sets_key] = [{"time_sec": 0}]
    else:
        if not st.session_state[sets_key] or "weight" not in st.session_state[sets_key][0]:
            st.session_state[sets_key] = [{"weight": 0, "reps": 0}]

    # Buttons add/remove sets (outside form is fine)
    c_plus, c_minus = st.columns([1, 1])
    with c_plus:
        if st.button("➕ Add set", key=f"{ns}_add_set_btn"):
            if profile["mode"] == "time":
                last_t = st.session_state[sets_key][-1].get("time_sec", 0) if st.session_state[sets_key] else 0
                st.session_state[sets_key].append({"time_sec": int(last_t)})
            else:
                last_w = st.session_state[sets_key][-1].get("weight", 0) if st.session_state[sets_key] else 0
                st.session_state[sets_key].append({"weight": int(last_w), "reps": 0})
            st.rerun()

    with c_minus:
        if st.button("➖ Remove last", key=f"{ns}_remove_set_btn"):
            if len(st.session_state[sets_key]) > 1:
                st.session_state[sets_key].pop()
                i = len(st.session_state[sets_key]) + 1
                st.session_state.pop(f"{ns}_w_{i}", None)
                st.session_state.pop(f"{ns}_r_{i}", None)
                st.session_state.pop(f"{ns}_t_{i}", None)
                st.rerun()

    # FORM: now Save is INSIDE the form -> it will capture latest widget values
    sets_rows: list[dict] = []
    with st.form(f"sets_form_{ns}", clear_on_submit=False):
        for idx, s in enumerate(st.session_state[sets_key], start=1):
            if profile["mode"] == "time":
                key_t = f"{ns}_t_{idx}"
                current_t = st.session_state.get(key_t, s.get("time_sec", 0))
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
                current_w = st.session_state.get(key_w, s.get("weight", 0))
                current_r = st.session_state.get(key_r, s.get("reps", 0))

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

        b1, b2 = st.columns([1, 1])
        with b1:
            apply = st.form_submit_button("✅ Apply sets")
        with b2:
            save = st.form_submit_button("💾 Save workout")

    # Apply updates local state (so summary reflects it)
    if apply or save:
        st.session_state[sets_key] = sets_rows

    # Summary
    st.markdown("### Session summary")
    if profile["mode"] == "time":
        filled = [s for s in st.session_state[sets_key] if s.get("time_sec", 0) > 0]
        st.info(f"Sets: {len(filled)} | Total time: {sum(s['time_sec'] for s in filled)} sec")
    else:
        filled = [s for s in st.session_state[sets_key] if s.get("weight", 0) > 0 and s.get("reps", 0) > 0]
        st.info(f"Sets: {len(filled)} | Total volume: {sum(s['weight'] * s['reps'] for s in filled)} kg")

    # Save logic (NOW uses the form values reliably)
    if save:
        try:
            if profile["mode"] == "time":
                cleaned = [s for s in st.session_state[sets_key] if s.get("time_sec", 0) > 0]
            else:
                cleaned = [s for s in st.session_state[sets_key] if s.get("weight", 0) > 0 and s.get("reps", 0) > 0]

            if not cleaned:
                st.error("Add at least one filled set.")
                st.stop()

            normalized = []
            for s in cleaned:
                if profile["mode"] == "time":
                    normalized.append({"weight": 0, "reps": 0, "time_sec": int(s["time_sec"])})
                else:
                    normalized.append({"weight": int(s["weight"]), "reps": int(s["reps"]), "time_sec": None})

            ex_id = add_exercise(engine, exercise_name)
            add_workout_with_sets(engine, str(workout_date), ex_id, normalized)

            st.success("Saved ✅")

            # reset sets for this exercise/date namespace
            st.session_state[sets_key] = [{"time_sec": 0}] if profile["mode"] == "time" else [{"weight": 0, "reps": 0}]
            for i in range(1, 50):
                st.session_state.pop(f"{ns}_w_{i}", None)
                st.session_state.pop(f"{ns}_r_{i}", None)
                st.session_state.pop(f"{ns}_t_{i}", None)

            st.rerun()

        except Exception as e:
            st.error(f"Save failed: {e}")

# ============================
# TAB: History
# ============================
with tab_history:
    st.subheader("History")

    hist = get_history_df()
    if hist.empty:
        st.info("No workouts yet.")
        st.stop()

    tmp = hist.copy()
    # vectorized-ish (still row-wise but ok for now)
    tmp["set_str"] = tmp.apply(
        lambda row: f"{int(row['time_sec'])}s" if pd.notna(row["time_sec"]) and int(row["time_sec"]) > 0
        else f"{int(row['weight'])}×{int(row['reps'])}",
        axis=1
    )

    compact = (
        tmp.sort_values(["workout_date", "workout_id", "set_no"])
           .groupby(["workout_id", "workout_date", "exercise"], as_index=False)
           .agg(sets=("set_str", lambda x: " | ".join(x)))
           .sort_values(["workout_date", "workout_id"], ascending=[False, False])
    )

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        ex_list = ["All"] + sorted(compact["exercise"].unique().tolist())
        ex_filter = st.selectbox("Exercise", ex_list, index=0, key="hist_ex_filter")
    with c2:
        dmin = pd.to_datetime(compact["workout_date"]).min().date()
        d_from = st.date_input("From", value=dmin, key="hist_from")
    with c3:
        dmax = pd.to_datetime(compact["workout_date"]).max().date()
        d_to = st.date_input("To", value=dmax, key="hist_to")

    view = compact.copy()
    view["workout_date_dt"] = pd.to_datetime(view["workout_date"]).dt.date
    view = view[(view["workout_date_dt"] >= d_from) & (view["workout_date_dt"] <= d_to)]
    if ex_filter != "All":
        view = view[view["exercise"] == ex_filter]

    if view.empty:
        st.info("No records for current filters.")
        st.stop()

    for day, day_df in view.groupby("workout_date", sort=False):
        day_df = day_df.sort_values("workout_id", ascending=False)
        with st.expander(f"📅 {day}  ·  {len(day_df)} entries", expanded=True):
            for _, row in day_df.iterrows():
                left, right = st.columns([5, 1])
                with left:
                    st.markdown(f"**{row['exercise']}**")
                    st.caption(row["sets"])
                with right:
                    pop = st.popover("🗑", use_container_width=True)
                    with pop:
                        st.write("Delete this entry?")
                        confirm = st.checkbox("Confirm", key=f"confirm_{row['workout_id']}")
                        if st.button("Delete", key=f"del_{row['workout_id']}", disabled=not confirm):
                            delete_workout(engine, int(row["workout_id"]))
                            st.success("Deleted ✅")
                            st.rerun()

# ============================
# TAB: Progress
# ============================
with tab_progress:
    st.subheader("Progress")

    t_hist = time.time()
    hist = get_history_df()
    dbg(f"history load: {time.time() - t_hist:.3f} sec")

    if hist.empty:
        st.info("Add workouts to see progress.")
        st.stop()

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
    dbg(f"calendar: {time.time() - t_cal:.3f} sec")

    trained_dates = sorted(
        [date(st.session_state.cal_year, st.session_state.cal_month, d) for d in entries_map.keys()],
        reverse=True
    )

    if trained_dates:
        pick_day = st.selectbox("Show workouts for day", trained_dates, key="cal_pick_day")

        # fetch day rows (dialect-safe)
        if engine.dialect.name == "postgresql":
            q = """
            SELECT
                w.id AS workout_id,
                w.workout_date::text AS workout_date,
                e.name AS exercise,
                s.set_no,
                s.weight,
                s.reps,
                s.time_sec
            FROM workouts w
            JOIN exercises e ON e.id = w.exercise_id
            JOIN sets s ON s.workout_id = w.id
            WHERE w.workout_date = :d::date
            ORDER BY w.id DESC, s.set_no ASC
            """
        else:
            q = """
            SELECT
                w.id AS workout_id,
                w.workout_date,
                e.name AS exercise,
                s.set_no,
                s.weight,
                s.reps,
                s.time_sec
            FROM workouts w
            JOIN exercises e ON e.id = w.exercise_id
            JOIN sets s ON s.workout_id = w.id
            WHERE w.workout_date = :d
            ORDER BY w.id DESC, s.set_no ASC
            """

        day_rows = pd.read_sql_query(text(q), engine, params={"d": str(pick_day)})

        if not day_rows.empty:
            day_rows["set_str"] = day_rows.apply(
                lambda row: f"{int(row['time_sec'])}s" if pd.notna(row["time_sec"]) and int(row["time_sec"]) > 0
                else f"{int(row['weight'])}×{int(row['reps'])}",
                axis=1
            )
            day_compact = (
                day_rows.groupby(["workout_id", "workout_date", "exercise"], as_index=False)
                        .agg(sets=("set_str", lambda x: " | ".join(x)))
                        .sort_values("workout_id", ascending=False)
            )

            st.markdown("### Workouts for selected day")
            for _, rr in day_compact.iterrows():
                st.markdown(f"**{rr['exercise']}**")
                st.caption(rr["sets"])
        else:
            st.info("No rows for this day (unexpected).")
    else:
        st.info("No workouts in this month yet.")

    st.divider()

    # ---- Exercise progress (1RM) ----
    non_timed = hist[(hist["time_sec"].isna()) | (hist["time_sec"].fillna(0) == 0)].copy()
    if non_timed.empty:
        st.info("Only timed exercises found (no weight/reps to calculate 1RM).")
        st.stop()

    exercises = sorted(non_timed["exercise"].unique().tolist())
    ex = st.selectbox("Exercise", exercises, key="progress_exercise_select")

    df = non_timed[(non_timed["exercise"] == ex) & (non_timed["weight"] > 0) & (non_timed["reps"] > 0)].copy()
    if df.empty:
        st.info("No valid weight+reps sets for this exercise.")
        st.stop()

    df["est_1rm"] = df["weight"] * (1 + (df["reps"] / 30.0))

    t_plot = time.time()
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

    dbg(f"plot: {time.time() - t_plot:.3f} sec")
    st.metric("🏆 Best estimated 1RM", f"{float(df['est_1rm'].max()):.1f}")

dbg(f"Render: {time.time() - start_total:.3f} sec")