import io
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import cv2

# -----------------------------
# Helpers
# -----------------------------
def init_state():
    defaults = {
        "answer_key_bytes": None,         # store bytes, not UploadedFile
        "student_papers_bytes": [],       # list[bytes]
        "results": None,                  # pandas DataFrame or None
        "debug_msgs": [],                 # debug log strings
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def log(msg: str):
    st.session_state.debug_msgs.append(msg)

def uploaded_file_to_bytes(uf) -> bytes:
    """Safely read UploadedFile to raw bytes (seek to start before reading)."""
    if uf is None:
        return None
    try:
        uf.seek(0)
    except Exception:
        pass
    return uf.read()

def bytes_to_cv2_image(b: bytes):
    """Decode bytes -> OpenCV BGR image."""
    if b is None:
        return None
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def bytes_to_pil_image(b: bytes):
    """Decode bytes -> PIL image (if you prefer PIL)."""
    if b is None:
        return None
    return Image.open(io.BytesIO(b)).convert("RGB")

# -----------------------------
# OMR detection adapter
# -----------------------------
# EXPECTED CONTRACT:
# omr_detect_answers(image, debug=False) -> Dict[int, str]
# returns mapping {question_index: "A"/"B"/"C"/"D"} (or whatever your schema is)
# IMPORTANT: Replace the stub below with your actual OMR function.
def omr_detect_answers(img_bgr, debug=False) -> Dict[int, str]:
    """
    Replace with your real OMR function.
    Return {} on failure to allow upstream error handling.
    """
    # Example stub: returns empty dict (forces visible error if not replaced)
    return {}

# -----------------------------
# Grading logic
# -----------------------------
def grade_student(key: Dict[int, str], stu: Dict[int, str]) -> Tuple[int, int, int, int, float]:
    """
    Returns: correct, wrong, blank, total, accuracy
    """
    if not key:
        return 0, 0, 0, 0, 0.0

    total = len(key)
    correct = 0
    wrong = 0
    blank = 0

    for q, ans in key.items():
        stu_ans = stu.get(q, None)
        if stu_ans is None or stu_ans == "":
            blank += 1
        elif stu_ans == ans:
            correct += 1
        else:
            wrong += 1

    accuracy = (correct / total) * 100 if total > 0 else 0.0
    return correct, wrong, blank, total, accuracy

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="OMR Answer Sheet Scanner & Grading System", layout="centered")
init_state()

st.markdown("## 📝 OMR Answer Sheet Scanner & Grading System")

with st.expander("Instructions", expanded=False):
    st.markdown(
        "- Upload the Answer Key image (JPG/PNG)\n"
        "- Upload one or more Student Answer Sheets\n"
        "- Click “Start Comparison”"
    )

# ---- Uploads
st.markdown("### 🔑 Upload Answer Key")
answer_key_upload = st.file_uploader(
    "Choose answer key image (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    key="answer_key_uploader",
)

if answer_key_upload is not None:
    st.session_state.answer_key_bytes = uploaded_file_to_bytes(answer_key_upload)

st.markdown("### 🎓 Upload Student Answer Sheets")
student_uploads = st.file_uploader(
    "Upload student answer sheets (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="student_uploader",
)

if student_uploads:
    # Overwrite with the fresh list of bytes every rerun
    st.session_state.student_papers_bytes = [uploaded_file_to_bytes(f) for f in student_uploads]

# ---- Quick status (counts)
num_students = len(st.session_state.student_papers_bytes or [])
key_present = st.session_state.answer_key_bytes is not None

col1, col2 = st.columns(2)
with col1:
    st.metric("Answer Key Uploaded", "Yes" if key_present else "No")
with col2:
    st.metric("Student Sheets Uploaded", str(num_students))

# ---- Actions
colA, colB = st.columns([2, 1])
with colA:
    start = st.button("🚦 Start Comparison", use_container_width=True)
with colB:
    if st.button("♻️ Reset", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.experimental_rerun()

# ---- Processing
if start:
    st.session_state.debug_msgs = []  # reset debug log
    if not key_present:
        st.error("Please upload the Answer Key image first.")
        st.stop()
    if num_students == 0:
        st.error("Please upload at least one Student Answer Sheet.")
        st.stop()

    log(f"Answer key bytes: {len(st.session_state.answer_key_bytes)}")
    log(f"Student papers count: {num_students}")

    with st.spinner("Running OMR detection..."):
        # Decode images
        key_img = bytes_to_cv2_image(st.session_state.answer_key_bytes)
        if key_img is None:
            st.error("Failed to decode the Answer Key image. Check the file format.")
            st.stop()

        key_answers = omr_detect_answers(key_img, debug=True)
        log(f"Key answers detected: {len(key_answers)}")

        if not key_answers:
            st.error("Failed to detect answers in the Answer Key. Please check alignment/quality.")
            st.stop()

        results_rows: List[dict] = []
        for i, b in enumerate(st.session_state.student_papers_bytes):
            stu_img = bytes_to_cv2_image(b)
            if stu_img is None:
                log(f"Student {i+1}: decode failed.")
                continue

            stu_answers = omr_detect_answers(stu_img, debug=True)
            if not stu_answers:
                log(f"Student {i+1}: OMR detection returned empty.")
                continue

            correct, wrong, blank, total, acc = grade_student(key_answers, stu_answers)

            results_rows.append({
                "Student": f"Student {i+1}",
                "Total Qs": total,
                "Correct": correct,
                "Wrong": wrong,
                "Blank": blank,
                "Score": correct,          # adjust if there is negative marking
                "Accuracy (%)": round(acc, 2),
            })

        if not results_rows:
            st.error("No valid student results. Check uploads or OMR detection.")
            st.session_state.results = pd.DataFrame()  # keep consistent type
        else:
            st.session_state.results = pd.DataFrame(results_rows)

# ---- Results display
st.markdown("---")
st.markdown("### 📊 Results")

if st.session_state.results is not None and not getattr(st.session_state.results, "empty", True):
    st.dataframe(st.session_state.results, use_container_width=True, hide_index=True)
    csv_bytes = st.session_state.results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV",
        data=csv_bytes,
        file_name="omr_results.csv",
        mime="text/csv",
    )
else:
    st.info("No results to display yet. Upload files and click ‘Start Comparison’.")

# ---- Optional debug
with st.expander("🧪 Debug Info"):
    st.write({
        "answer_key_present": key_present,
        "student_papers_count": num_students,
        "results_shape": None if st.session_state.results is None else st.session_state.results.shape,
    })
    if st.session_state.debug_msgs:
        st.code("\n".join(st.session_state.debug_msgs))
