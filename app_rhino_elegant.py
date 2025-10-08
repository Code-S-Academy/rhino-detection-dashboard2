import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ==========================
# Page config & Global Style
# ==========================
st.set_page_config(page_title="🦏 Rhino Detection Dashboard", page_icon="🦏", layout="wide")

# ---- Theme + Glass UI ----
st.markdown(
    """
    <style>
      :root {
        --bg: #f5f7fb;
        --card: #ffffff;
        --card-border: #e5e7eb;
        --text: #111827;
        --muted: #6b7280;
        --accent: #3b82f6;
        --accent-2: #10b981;
        --danger: #ef4444;
        --warn: #f59e0b;
        --ok: #16a34a;
      }
      html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg);
        color: var(--text);
      }
      #MainMenu {visibility: hidden;} footer {visibility: hidden;}

      .hero { border: 1px solid var(--card-border);
              background: linear-gradient(120deg, rgba(59,130,246,.08), rgba(16,185,129,.08));
              border-radius: 18px; padding: 18px 20px; }
      .hero h1 { margin: 0 0 6px 0; font-size: 1.9rem; letter-spacing: .2px;}
      .hero p { margin: 0; color: var(--muted); }

      .glass { background: var(--card); border: 1px solid var(--card-border); border-radius: 14px; padding: 14px 16px;
               box-shadow: 0 4px 12px rgba(0,0,0,.05); }
      .kpi-label {font-size: .8rem; color: var(--muted);} 
      .kpi-value {font-size: 1.6rem; font-weight: 800; letter-spacing: .3px; color: var(--text);}

      .pill {display:inline-flex; align-items:center; gap:.4rem; padding:.25rem .6rem; border-radius:999px;
             border:1px solid var(--card-border); background: #f9fafb; font-size:.8rem;}
      .dot {width:.55rem; height:.55rem; border-radius:50%}
      .dot.ok {background: var(--ok)} .dot.warn {background: var(--warn)} .dot.danger {background: var(--danger)}

      .stDataFrame table thead tr th {font-weight: 700 !important; color: var(--text) !important;}
      details > summary {font-weight: 700;}
      section[data-testid="stSidebar"] {background: #ffffff; border-right: 1px solid var(--card-border);}
    </style>
    """, unsafe_allow_html=True
)

# ==========================
# Sidebar Controls
# ==========================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    data_dir = st.text_input("Outputs folder path", value="./outputs_fbf")
    annotated_video_name = st.text_input("Annotated video filename", value="rhino_annotated_fbf.mp4")

    st.markdown("---")
    st.markdown("#### 🧪 Data Fallbacks")
    enable_dummy = st.toggle("Auto-create Camera & Patrol if missing/empty", value=True)
    dummy_rows = st.slider("Dummy rows", 25, 300, 120, 5)
    rand_seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    st.markdown("---")
    st.markdown("#### 🎬 Video Options")
    bytes_mode = st.checkbox("Compatibility mode (load video as bytes)", value=False)
    fallback_opencv = st.checkbox("Fallback: OpenCV frame viewer", value=False)

    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        reload_btn = st.button("📥 Reload data", type="primary")
    with colB:
        write_dummy_btn = st.button("💾 Write Camera/Patrol CSVs")

# ==========================
# Paths & IO
# ==========================
def make_paths(base_dir: str, annotated_name: str):
    base = Path(base_dir)
    return {
        "camera": base / "camera_dataset.csv",
        "patrol": base / "patrol_dataset.csv",
        "detection": base / "detection_dataset.csv",
        "threat": base / "threat_assessment_dataset.csv",
        "prediction": base / "prediction_dataset.csv",
        "video": base / annotated_name,
        "base": base,
    }

def safe_read_csv(p: Path):
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception as e:
        st.warning(f"Failed to read {p.name}: {e}")
    return None

# ==========================
# Dummy Generators
# ==========================
def dummy_camera(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    frames = np.arange(n)
    return pd.DataFrame({
        "frame_index": frames,
        "cam_id": rng.integers(1, 4, size=n),
        "lat": -1.95 + rng.normal(0, 0.001, size=n),
        "lon": 30.06 + rng.normal(0, 0.001, size=n),
        "timestamp_sec": np.round(frames / 5.0, 2),
    })

def dummy_patrol(n: int, seed: int = 42):
    rng = np.random.default_rng(seed + 1)
    frames = np.arange(n)
    rows = []
    for f in frames:
        if rng.uniform() < 0.18:
            rows.append([f, int(rng.integers(100, 160)), float(rng.uniform(8, 160))])
    return pd.DataFrame(rows, columns=["frame_index", "person_id", "distance_m"])

# ==========================
# Helpers
# ==========================
def frames_processed(cam_df, det_df, thr_df, video_path: Path):
    if cam_df is not None and not cam_df.empty and "frame_index" in cam_df.columns:
        return int(cam_df["frame_index"].nunique())
    for df in (det_df, thr_df):
        if df is not None and not df.empty and "frame_index" in df.columns:
            return int(df["frame_index"].nunique())
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    except Exception:
        return 0

def unique_rhinos(det_df, thr_df):
    if thr_df is not None and not thr_df.empty and "rhino_id" in thr_df.columns:
        return int(thr_df["rhino_id"].nunique())
    if det_df is not None and not det_df.empty and "rhino_id" in det_df.columns:
        return int(det_df["rhino_id"].nunique())
    if det_df is not None and not det_df.empty and "frame_index" in det_df.columns:
        return int(det_df.groupby("frame_index").size().max())
    return 0

def risk_counts(thr_df):
    if thr_df is None or thr_df.empty or "risk_level" not in thr_df.columns:
        return 0,0,0
    c = thr_df["risk_level"].value_counts().to_dict()
    return int(c.get("HIGH",0)), int(c.get("MEDIUM",0)), int(c.get("LOW",0))

def fix_prediction_df(pred_df: pd.DataFrame, det_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Ensure a 'poaching_prob' column exists. Avoid length mismatches by rebuilding a slim DF when needed."""
    if pred_df is None or pred_df.empty:
        return pred_df
    cols = list(pred_df.columns)
    if "poaching_prob" in cols:
        return pred_df

    # Map common alternatives
    alt_prob = next((c for c in ["poaching_probability","probability","risk_prob","prob","score","confidence","conf"] if c in cols), None)
    fixed = pred_df.copy()

    if alt_prob is not None:
        fixed[alt_prob] = pd.to_numeric(fixed[alt_prob], errors="coerce")
        maxv = fixed[alt_prob].max()
        fixed["poaching_prob"] = (fixed[alt_prob] / 100.0).clip(0,1) if (pd.notna(maxv) and maxv > 1.0) else fixed[alt_prob].clip(0,1)
        return fixed

    # No usable column: rebuild a minimal DF whose length matches the IDs list to avoid ValueError
    rng = np.random.default_rng(seed)
    if "rhino_id" in fixed.columns and fixed["rhino_id"].notna().any():
        ids = fixed["rhino_id"].dropna().unique().tolist()
        fixed = pd.DataFrame({"rhino_id": ids})
    elif det_df is not None and "rhino_id" in det_df.columns and len(det_df) > 0:
        ids = det_df["rhino_id"].dropna().unique().tolist()
        fixed = pd.DataFrame({"rhino_id": ids})
    else:
        fixed = pd.DataFrame({"rhino_id": [1,2,3]})

    n = len(fixed)
    fixed["poaching_prob"] = rng.uniform(0.1, 0.9, size=n)
    if "eta_sec" not in fixed.columns:
        fixed["eta_sec"] = rng.uniform(10, 180, size=n).round(2)
    return fixed

# ==========================
# Load Data
# ==========================
P = make_paths(data_dir, annotated_video_name)
if "cache_v4" not in st.session_state:
    st.session_state["cache_v4"] = {}

if reload_btn or not st.session_state["cache_v4"]:
    cam_df = safe_read_csv(P["camera"])
    pat_df = safe_read_csv(P["patrol"])
    det_df = safe_read_csv(P["detection"])
    thr_df = safe_read_csv(P["threat"])
    pred_df = safe_read_csv(P["prediction"])

    if enable_dummy:
        if cam_df is None or cam_df.empty:
            cam_df = dummy_camera(int(dummy_rows), rand_seed)
        if pat_df is None or pat_df.empty:
            pat_df = dummy_patrol(int(dummy_rows), rand_seed)

    pred_df = fix_prediction_df(pred_df, det_df, seed=rand_seed)

    st.session_state["cache_v4"] = dict(camera=cam_df, patrol=pat_df, detection=det_df, threat=thr_df, prediction=pred_df)

cam_df = st.session_state["cache_v4"]["camera"]
pat_df = st.session_state["cache_v4"]["patrol"]
det_df = st.session_state["cache_v4"]["detection"]
thr_df = st.session_state["cache_v4"]["threat"]
pred_df = st.session_state["cache_v4"]["prediction"]
video_path = P["video"]

# Optionally write current Camera & Patrol
if write_dummy_btn:
    P["base"].mkdir(parents=True, exist_ok=True)
    if cam_df is not None and not cam_df.empty:
        cam_df.to_csv(P["camera"], index=False)
    if pat_df is not None and not pat_df.empty:
        pat_df.to_csv(P["patrol"], index=False)
    st.success("Camera & Patrol CSVs written.")

# ==========================
# HERO
# ==========================
st.markdown(
    """
    <div class="hero">
      <h1>🦏 Rhino Detection Dashboard</h1>
      <p>Intelligence powered by AI</p>
    </div>
    """, unsafe_allow_html=True
)

# ==========================
# KPI STRIP
# ==========================
col1, col2, col3, col4, col5 = st.columns(5, gap="small")
with col1:
    st.markdown('<div class="glass"><div class="kpi-label">Frames Processed</div><div class="kpi-value">{} </div></div>'.format(frames_processed(cam_df, det_df, thr_df, video_path)), unsafe_allow_html=True)
with col2:
    st.markdown('<div class="glass"><div class="kpi-label">Unique Rhinos</div><div class="kpi-value">{} </div></div>'.format(unique_rhinos(det_df, thr_df)), unsafe_allow_html=True)
with col3:
    st.markdown('<div class="glass"><div class="kpi-label">Rhino Detections</div><div class="kpi-value">{} </div></div>'.format(0 if det_df is None else len(det_df)), unsafe_allow_html=True)
with col4:
    st.markdown('<div class="glass"><div class="kpi-label">Patrol Sightings</div><div class="kpi-value">{} </div></div>'.format(0 if pat_df is None else len(pat_df)), unsafe_allow_html=True)
with col5:
    h, m, l = risk_counts(thr_df)
    st.markdown('<div class="glass"><div class="kpi-label">Risk (H/M/L)</div><div class="kpi-value">{}/{}/{} </div></div>'.format(h, m, l), unsafe_allow_html=True)

# ==========================
# TABS
# ==========================
t_over, t_charts, t_data, t_video = st.tabs(["Overview", "Charts", "Datasets", "Video"])

with t_over:
    left, right = st.columns([1.8, 1], gap="large")
    with left:
        st.subheader("System Status")
        total = sum(risk_counts(thr_df))
        status = ("ok","Nominal")
        if h > 0 or m > max(1, int(0.25*max(total,1))):
            status = ("warn","Watch")
        if h > max(2, int(0.15*max(total,1))):
            status = ("danger","Critical")
        st.markdown(f"<span class='pill'><span class='dot {status[0]}'></span><b>{status[1]}</b></span>", unsafe_allow_html=True)
        st.caption("Derived from threat risk distribution.")
        st.markdown("##### Notes")
        st.write(f"• Data folder: `{data_dir}`")
        st.write(f"• Video file: `{annotated_video_name}`")
        # st.write(f"• Camera/Patrol fallback: **{'ON' if enable_dummy else 'OFF'}**")
    with right:
        st.subheader("Prediction Snapshot")
        if pred_df is None or pred_df.empty:
            st.info("No prediction data available.")
        else:
            tmp = pred_df.copy()
            if "poaching_prob" in tmp.columns:
                tmp["poaching_prob"] = pd.to_numeric(tmp["poaching_prob"], errors="coerce")
                tmp = tmp.dropna(subset=["poaching_prob"])
                top = tmp.sort_values("poaching_prob", ascending=False).head(5).copy()
                if len(top) and top["poaching_prob"].max() <= 1:
                    top["poaching_prob"] = (top["poaching_prob"]*100).round(1).astype(str) + "%"
                st.dataframe(top, use_container_width=True, hide_index=True)
            else:
                st.dataframe(tmp.head(), use_container_width=True, hide_index=True)

with t_charts:
    c1, c2 = st.columns([1.8, 1], gap="large")
    with c1:
        st.subheader("Risk over time")
        if thr_df is None or thr_df.empty or "timestamp_sec" not in thr_df.columns:
            st.info("No threat data found.")
        else:
            tdf = thr_df.copy()
            risk_map = {"LOW":0, "MEDIUM":1, "HIGH":2}
            tdf["risk_num"] = tdf["risk_level"].map(risk_map).fillna(0).astype(int)
            min_t, max_t = float(tdf["timestamp_sec"].min()), float(tdf["timestamp_sec"].max())
            step = max((max_t - min_t) / 100, 0.01)
            rng = st.slider("Time range (sec)", min_value=min_t, max_value=max_t, value=(min_t, max_t), step=step)
            allowed = st.multiselect("Risk levels", ["LOW","MEDIUM","HIGH"], default=["LOW","MEDIUM","HIGH"], label_visibility="collapsed")
            mask = (tdf["timestamp_sec"] >= rng[0]) & (tdf["timestamp_sec"] <= rng[1]) & (tdf["risk_level"].isin(allowed))
            fdf = tdf.loc[mask].sort_values("timestamp_sec")
            st.line_chart(fdf.set_index("timestamp_sec")["risk_num"], height=260)
            st.caption("Scale: 0=LOW, 1=MEDIUM, 2=HIGH")
    with c2:
        st.subheader("Risk distribution")
        if thr_df is None or thr_df.empty or "risk_level" not in thr_df.columns:
            st.info("No data")
        else:
            counts = thr_df["risk_level"].value_counts().reindex(["LOW","MEDIUM","HIGH"]).fillna(0).astype(int)
            st.bar_chart(counts, height=260)

with t_data:
    st.subheader("Tables & Downloads")
    def table_download(title, df):
        with st.expander(title, expanded=(title in ["camera_dataset","patrol_dataset","prediction_dataset"])):
            if df is None or df.empty:
                st.info("No data")
                return
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(f"⬇️ Download {title}.csv", df.to_csv(index=False).encode("utf-8"),
                               file_name=f"{title}.csv", mime="text/csv")
    table_download("camera_dataset", cam_df)
    table_download("patrol_dataset", pat_df)
    table_download("detection_dataset", det_df)
    table_download("threat_assessment_dataset", thr_df)
    table_download("prediction_dataset", pred_df)

with t_video:
    st.subheader("Annotated video preview")
    if not video_path:
        st.info("No video path set.")
    else:
        exists = video_path.exists()
        size_mb = (video_path.stat().st_size / (1024*1024)) if exists else 0
        st.write(f"**Path:** {video_path}")
        st.write(f"**Exists:** {exists} • **Size:** {size_mb:.2f} MB")
        if exists and size_mb > 0:
            ext = video_path.suffix.lower()
            if not fallback_opencv:
                try:
                    if bytes_mode:
                        data = video_path.read_bytes()
                        fmt = "video/mp4" if ext == ".mp4" else ("video/webm" if ext == ".webm" else "video/x-msvideo")
                        st.video(data, format=fmt, start_time=0)
                        st.caption("Loaded in compatibility (bytes) mode.")
                    else:
                        st.video(str(video_path))
                        st.caption("Loaded by file path.")
                except Exception as e:
                    st.error(f"Failed to load video: {e}")
                    st.info("Try 'Fallback: OpenCV frame viewer' or re-encode to H.264 +faststart.")
            else:
                try:
                    import cv2
                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        st.error("OpenCV could not open the video file.")
                    else:
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                        duration = total_frames / fps if fps > 0 else 0
                        st.write(f"Frames: {total_frames} • FPS: {fps:.2f} • Duration: {duration:.2f}s")
                        t = st.slider("Preview time (s)", 0.0, max(0.1, float(duration)), 0.0, 0.1, key="fb_t_v4")
                        cap.set(cv2.CAP_PROP_POS_MSEC, t*1000.0)
                        ok, frame = cap.read()
                        cap.release()
                        if ok:
                            frame_rgb = frame[:, :, ::-1]
                            st.image(frame_rgb, caption=f"Frame at {t:.2f}s", use_container_width=True)
                        else:
                            st.warning("Could not read frame at that time.")
                except Exception as e:
                    st.error(f"OpenCV fallback failed: {e}")

st.caption("Developed by Code-S-Academy")
