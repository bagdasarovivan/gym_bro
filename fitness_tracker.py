import streamlit as st
import calendar
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from sqlalchemy import create_engine, text


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
    "Lunges": "light",
    "Leg Curl": "light",
    "Leg Extension": "light",
    "Push-Ups": "light",
    "Pull-Ups": "light",
    "Crunches": "light",
    "Flat Dumbbell Flyes": "light",

    # timed
    "Plank": "timed",

    # if you keep them as exercises
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
}

# ----------------- DB helpers -----------------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

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

    conn.commit()

def get_exercises(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT id, name FROM exercises ORDER BY name", conn)

def add_exercise(conn, name: str) -> int:
    name = name.strip()
    if not name:
        raise ValueError("Empty exercise name")
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO exercises(name) VALUES(?)", (name,))
    conn.commit()
    row = cur.execute("SELECT id FROM exercises WHERE name = ?", (name,)).fetchone()
    return int(row[0])

def add_workout_with_sets(conn, workout_date: str, exercise_id: int, sets_rows: list[dict]):
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO workouts(workout_date, exercise_id) VALUES(?, ?)",
        (workout_date, exercise_id)
    )
    workout_id = cur.lastrowid

    for i, s in enumerate(sets_rows, start=1):
        cur.execute("""
            INSERT INTO sets(workout_id, set_no, weight, reps, time_sec)
            VALUES(?, ?, ?, ?, ?)
        """, (
            workout_id,
            i,
            float(s.get("weight", 0)),
            int(s.get("reps", 0)),
            int(s.get("time_sec")) if s.get("time_sec") is not None else None
        ))

    conn.commit()

def get_history(conn) -> pd.DataFrame:
    return pd.read_sql_query("""
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
    """, conn)

def get_last_workout_for_exercise(conn, exercise_name: str):
    df = pd.read_sql_query("""
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
        WHERE e.name = ?
        ORDER BY w.workout_date DESC, w.id DESC, s.set_no ASC
    """, conn, params=(exercise_name,))

    if df.empty:
        return None

    last_workout_id = int(df.iloc[0]["workout_id"])
    return df[df["workout_id"] == last_workout_id].copy()

def delete_workout(conn, workout_id: int):
    cur = conn.cursor()
    cur.execute("DELETE FROM sets WHERE workout_id = ?", (workout_id,))
    cur.execute("DELETE FROM workouts WHERE id = ?", (workout_id,))
    conn.commit()

# ----------------- App -----------------
st.set_page_config(
    page_title="Gym BRO",
    page_icon="images/gymbro_icon.png",
    layout="centered"
)

# --- Supabase connection test (temporary) ---
if "DB_URL" in st.secrets:
    try:
        engine = create_engine(st.secrets["DB_URL"])
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        st.success("✅ Supabase DB connected")
    except Exception as e:
        st.error(f"❌ Supabase DB connect failed: {e}")
# -------------------------------------------


st.markdown("""
<style>

/* Мобильная адаптация */
@media (max-width: 768px) {

    h1 {
        font-size: 32px !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 40px !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 16px !important;
        padding: 12px 0px !important;
    }

    img {
        max-width: 100% !important;
        height: auto !important;
    }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>

/* Центрируем табы и задаём расстояние */
.stTabs [data-baseweb="tab-list"] {
    width: 100%;
    justify-content: center;
    gap: 100px;
}

/* Делаем вкладки крупнее */
.stTabs [data-baseweb="tab"] {
    font-size: 24px;          /* было меньше */
    padding: 20px 0px;        /* увеличиваем высоту */
    font-weight: 500;
}

/* Активная вкладка чуть жирнее */
.stTabs [aria-selected="true"] {
    font-weight: 700;
}

/* Немного мягкого hover-эффекта */
.stTabs [data-baseweb="tab"]:hover {
    opacity: 0.8;
}

</style>
""", unsafe_allow_html=True)


col1, col2 = st.columns([1, 6])
with col1:
    st.image("images/gymbro_logo.png", width=90)
with col2:
    st.markdown("<h1 style='margin:0'>Gym BRO</h1>", unsafe_allow_html=True)

conn = get_conn()
init_db(conn)

# seed exercises
for name in [
    "Bench Press","Squat","Deadlift","Biceps","Triceps","Overhead Press",
    "Dumbbell Flyes","Romanian Deadlift","Incline Dumbbell Press","Lat Pulldown",
    "Seated Cable Row","Dumbbell Bench","Push-Ups","Leg Press","Lunges","Leg Curl",
    "Leg Extension","Barbell Row","Plank","Pull-Ups","Crunches"
]:
    add_exercise(conn, name)

tab_add, tab_history, tab_progress = st.tabs(["➕ Add workout", "📜 History", "📈 Progress"])

# ----------------- TAB: Add workout -----------------
with tab_add:
    st.subheader("Add workout")

    ex_df = get_exercises(conn)
    ex_names = ex_df["name"].tolist()
    if not ex_names:
        st.warning("No exercises found.")
        st.stop()

    date_col, spacer = st.columns([1, 4])
    with date_col:
        workout_date = st.date_input(
            "📅 Date",
            value=date.today(),
            min_value=date(2020, 1, 1),
            max_value=date.today(),
            key="workout_date"
        )

    with st.expander("🔎 Search", expanded=False):
        q = st.text_input("Search exercise", value="", key="exercise_search")

    filtered = [x for x in ex_names if q.lower() in x.lower()] if q else ex_names

    fav = [x for x in FAVORITE_EXERCISES if x in filtered]
    rest = [x for x in filtered if x not in fav]
    ex_options = fav + rest

    if not ex_options:
        st.info("No exercises match your search.")
        st.stop()

    cA, cB = st.columns([3, 1])
    with cA:
        exercise_name_to_use = st.selectbox("Exercise", ex_options, key="add_exercise_select")
        
    if not exercise_name_to_use:
        st.stop()

    with cB:
        img_path = EXERCISE_IMAGES.get(exercise_name_to_use)
        if img_path:
            st.image(img_path, width=120)

    # --- Last workout (does NOT control whether Sets render) ---
    last_df = get_last_workout_for_exercise(conn, exercise_name_to_use)
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

    # ---- profile for current exercise ----
    ex_type = EXERCISE_TYPE.get(exercise_name_to_use, "light")
    profile = TYPE_PROFILES[ex_type]

    st.markdown("### Sets")

    # init sets for current mode
    if "sets" not in st.session_state:
        st.session_state.sets = [{"time_sec": 0}] if profile["mode"] == "time" else [{"weight": 0, "reps": 0}]

    # if user switches exercise type, re-init correctly
    if profile["mode"] == "time":
        if not st.session_state.sets or "time_sec" not in st.session_state.sets[0]:
            st.session_state.sets = [{"time_sec": 0}]
    else:
        if not st.session_state.sets or "weight" not in st.session_state.sets[0]:
            st.session_state.sets = [{"weight": 0, "reps": 0}]

    sets_rows = []

    for idx, s in enumerate(st.session_state.sets, start=1):
        if profile["mode"] == "time":
            key_t = f"t_{idx}"
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

            key_w = f"w_{idx}"
            key_r = f"r_{idx}"

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

    st.session_state.sets = sets_rows

    # --- Add / Remove set buttons ---
    c_plus, c_minus = st.columns([1, 1])

    with c_plus:
        if st.button("➕ Add set", key="add_set_btn"):
            if profile["mode"] == "time":
                last_t = st.session_state.sets[-1].get("time_sec", 0) if st.session_state.sets else 0
                st.session_state.sets.append({"time_sec": int(last_t)})
            else:
                last_w = st.session_state.sets[-1].get("weight", 0) if st.session_state.sets else 0
                st.session_state.sets.append({"weight": int(last_w), "reps": 0})
            st.rerun()

    with c_minus:
        if st.button("➖ Remove last", key="remove_set_btn"):
            if len(st.session_state.sets) > 1:
                st.session_state.sets.pop()
                i = len(st.session_state.sets) + 1
                st.session_state.pop(f"w_{i}", None)
                st.session_state.pop(f"r_{i}", None)
                st.session_state.pop(f"t_{i}", None)
                st.rerun()

    # ----------------- Session summary -----------------
    st.markdown("### Session summary")

    if profile["mode"] == "time":
        filled = [s for s in st.session_state.sets if s.get("time_sec", 0) > 0]
        total_sets = len(filled)
        total_time = sum(s["time_sec"] for s in filled)
        st.info(f"Sets: {total_sets} | Total time: {total_time} sec")
    else:
        filled = [s for s in st.session_state.sets if s.get("weight", 0) > 0 and s.get("reps", 0) > 0]
        total_sets = len(filled)
        total_volume = sum(s["weight"] * s["reps"] for s in filled)
        st.info(f"Sets: {total_sets} | Total volume: {total_volume} kg")

    # ----------------- Save -----------------
    if st.button("💾 Save workout"):
        try:
            if profile["mode"] == "time":
                cleaned = [s for s in st.session_state.sets if s.get("time_sec", 0) > 0]
            else:
                cleaned = [s for s in st.session_state.sets if s.get("weight", 0) > 0 and s.get("reps", 0) > 0]

            if not cleaned:
                st.error("Add at least one filled set.")
                st.stop()

            normalized = []
            for s in cleaned:
                if profile["mode"] == "time":
                    normalized.append({"weight": 0, "reps": 0, "time_sec": int(s["time_sec"])})
                else:
                    normalized.append({"weight": int(s["weight"]), "reps": int(s["reps"]), "time_sec": None})

            ex_id = add_exercise(conn, exercise_name_to_use)
            add_workout_with_sets(conn, str(workout_date), ex_id, normalized)

            st.success("Saved ✅")

            st.session_state.sets = [{"time_sec": 0}] if profile["mode"] == "time" else [{"weight": 0, "reps": 0}]

            for i in range(1, 50):
                st.session_state.pop(f"w_{i}", None)
                st.session_state.pop(f"r_{i}", None)
                st.session_state.pop(f"t_{i}", None)

            st.rerun()

        except Exception as e:
            st.error(f"Save failed: {e}")

# ----------------- TAB: History (with delete) -----------------
with tab_history:
    st.subheader("History")

    hist = get_history(conn)
    if hist.empty:
        st.info("No workouts yet.")
        st.stop()

    tmp = hist.copy()
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

    # -------- Filters ----------
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        ex_list = ["All"] + sorted(compact["exercise"].unique().tolist())
        ex_filter = st.selectbox("Exercise", ex_list, index=0)
    with c2:
        dmin = pd.to_datetime(compact["workout_date"]).min().date()
        d_from = st.date_input("From", value=dmin)
    with c3:
        dmax = pd.to_datetime(compact["workout_date"]).max().date()
        d_to = st.date_input("To", value=dmax)

    view = compact.copy()
    view["workout_date_dt"] = pd.to_datetime(view["workout_date"]).dt.date
    view = view[(view["workout_date_dt"] >= d_from) & (view["workout_date_dt"] <= d_to)]
    if ex_filter != "All":
        view = view[view["exercise"] == ex_filter]

    if view.empty:
        st.info("No records for current filters.")
        st.stop()

    # -------- Diary view (group by date) ----------
    for day, day_df in view.groupby("workout_date", sort=False):
        day_df = day_df.sort_values("workout_id", ascending=False)

        with st.expander(f"📅 {day}  ·  {len(day_df)} entries", expanded=True):
            for _, row in day_df.iterrows():
                left, right = st.columns([5, 1])

                with left:
                    st.markdown(f"**{row['exercise']}**")
                    st.caption(row["sets"])

                with right:
                    # safer delete with confirmation
                    pop = st.popover("🗑", use_container_width=True)
                    with pop:
                        st.write("Delete this entry?")
                        confirm = st.checkbox("Confirm", key=f"confirm_{row['workout_id']}")
                        if st.button("Delete", key=f"del_{row['workout_id']}", disabled=not confirm):
                            delete_workout(conn, int(row["workout_id"]))
                            st.success("Deleted ✅")
                            st.rerun()

   


# ----------------- TAB: Progress -----------------
with tab_progress:
    st.subheader("Progress")

    hist = get_history(conn)
    if hist.empty:
        st.info("Add workouts to see progress.")
        st.stop()


    st.markdown("## 🗓 Training calendar")

    # --- month navigation state ---
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

    # --- fetch training days for this month ---
    ym_start = f"{st.session_state.cal_year:04d}-{st.session_state.cal_month:02d}-01"
    last_day = calendar.monthrange(st.session_state.cal_year, st.session_state.cal_month)[1]
    ym_end = f"{st.session_state.cal_year:04d}-{st.session_state.cal_month:02d}-{last_day:02d}"

    month_daily = pd.read_sql_query("""
    SELECT workout_date, COUNT(*) AS entries
    FROM workouts
    WHERE workout_date BETWEEN ? AND ?
    GROUP BY workout_date
    """, conn, params=(ym_start, ym_end))

    entries_map = {}
    if not month_daily.empty:
        for _, r in month_daily.iterrows():
            d = pd.to_datetime(r["workout_date"]).date().day
            entries_map[int(d)] = int(r["entries"])

    # --- render calendar grid as HTML ---
    cal = calendar.Calendar(firstweekday=0)  # Monday
    weeks = cal.monthdayscalendar(st.session_state.cal_year, st.session_state.cal_month)

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

    # --- drilldown by selected day (simple) ---
    trained_dates = sorted(
        [date(st.session_state.cal_year, st.session_state.cal_month, d) for d in entries_map.keys()],
        reverse=True
    )
    if trained_dates:
        pick_day = st.selectbox("Show workouts for day", trained_dates, key="cal_pick_day")
        # твой existing day drilldown (как раньше)
    else:
        st.info("No workouts in this month yet.")

    # =======================
    # 🗓 Training calendar
    # =======================

        trained_days = cal[cal["entries"] > 0]["day"].tolist()[::-1]
        if trained_days:
            pick = st.selectbox("Show workouts for day", trained_days)

            day_rows = pd.read_sql_query("""
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
                WHERE w.workout_date = ?
                ORDER BY w.id DESC, s.set_no ASC
            """, conn, params=(str(pick),))

            if day_rows.empty:
                st.info("No rows for this day (unexpected).")
            else:
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

                for _, rr in day_compact.iterrows():
                    st.markdown(f"**{rr['exercise']}**")
                    st.caption(rr["sets"])
        else:
            st.info("No training days in the selected range yet.")

    st.divider()

    # =======================
    # 📈 Exercise progress (1RM)
    # =======================
    non_timed = hist[(hist["time_sec"].isna()) | (hist["time_sec"].fillna(0) == 0)].copy()

    if non_timed.empty:
        st.info("Only timed exercises found (no weight/reps to calculate 1RM).")
        st.stop()

    exercises = sorted(non_timed["exercise"].unique().tolist())
    ex = st.selectbox("Exercise", exercises, key="progress_exercise_select")

    df = non_timed[non_timed["exercise"] == ex].copy()
    df = df[(df["weight"] > 0) & (df["reps"] > 0)]

    if df.empty:
        st.info("No valid weight+reps sets for this exercise.")
        st.stop()

    # --- metrics by set ---
    df["est_1rm"] = df["weight"] * (1 + (df["reps"] / 30.0))

    # --- daily series ---
    best_1rm_by_day = df.groupby("workout_date", as_index=False)["est_1rm"].max()
    top_w_by_day = df.groupby("workout_date", as_index=False)["weight"].max()

    # --- plot: two axes ---
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

    # --- single metric ---
    st.metric("🏆 Best estimated 1RM", f"{float(df['est_1rm'].max()):.1f}")
