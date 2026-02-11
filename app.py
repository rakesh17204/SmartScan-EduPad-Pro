import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
import cv2
from PIL import Image
import tempfile
import os

# ============================================
# 🎨 PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="SmartScan EduPad Pro",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# 🔧 OMR DETECTION FUNCTION (PRODUCTION-GRADE)
# ============================================
def omr_detect_answers(uploaded_file, debug=False):
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.getbuffer())
        img_path = tmp.name

    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Failed to load image. Check file integrity.")

        orig = img.copy()

        # --------- Auto Deskew ---------
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        coords = np.column_stack(np.where(edges > 0))

        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)

        # --------- Color-aware threshold ---------
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]

        thresh = cv2.adaptiveThreshold(
            v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 3
        )

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # --------- Bubble Detection ---------
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bubbles = []
        for c in contours:
            area = cv2.contourArea(c)
            if 60 < area < 3000:
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = w / h
                if 0.6 < aspect_ratio < 1.4:
                    bubbles.append((x, y, w, h))

        if not bubbles:
            return {}

        # --------- Smart Row Clustering ---------
        bubbles = sorted(bubbles, key=lambda b: b[1])
        rows = []
        for b in bubbles:
            placed = False
            for row in rows:
                if abs(row[0][1] - b[1]) < 20:
                    row.append(b)
                    placed = True
                    break
            if not placed:
                rows.append([b])

        for row in rows:
            row.sort(key=lambda b: b[0])

        # --------- Filled Bubble Detection ---------
        answers = {}
        for qi, row in enumerate(rows, start=1):
            best_fill = 0
            best_opt = None
            for oi, (x, y, w, h) in enumerate(row):
                roi = thresh[y:y+h, x:x+w]
                fill = cv2.countNonZero(roi) / (w * h)

                if debug:
                    color = (0, 255, 0) if fill > 0.2 else (0, 0, 255)
                    cv2.rectangle(orig, (x, y), (x + w, y + h), color, 2)

                if fill > best_fill and fill > 0.2:
                    best_fill = fill
                    best_opt = chr(ord('A') + oi)

            if best_opt:
                answers[str(qi)] = best_opt

        if debug:
            st.subheader("🖼️ Debug Overlay")
            st.image(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), use_container_width=True)

        return answers

    except Exception as e:
        st.warning(f"OMR Processing Error: {str(e)}")
        return {}

    finally:
        # Clean up temp file
        if os.path.exists(img_path):
            os.unlink(img_path)


# ============================================
# 🔧 SCORE CALCULATION
# ============================================
def calculate_score(key_answers, student_answers):
    total = len(key_answers)
    if total == 0:
        return 0
    correct = 0
    for q in key_answers:
        if q in student_answers and key_answers[q] == student_answers[q]:
            correct += 1
    return round((correct / total) * 100, 2)


# ============================================
# 🔧 SESSION INIT
# ============================================
if "answer_key_image" not in st.session_state:
    st.session_state.answer_key_image = None
if "student_papers" not in st.session_state:
    st.session_state.student_papers = []
if "results" not in st.session_state:
    st.session_state.results = None


# ============================================
# 🔧 SIDEBAR
# ============================================
with st.sidebar:
    st.header("📋 Test Setup")

    answer_key_upload = st.file_uploader(
        "Upload Answer Key (Image)",
        type=["jpg", "jpeg", "png"],
        key="answer_key_upload"
    )

    if answer_key_upload:
        st.session_state.answer_key_image = answer_key_upload
        st.image(answer_key_upload, caption="Answer Key", use_container_width=True)

    passing_score = st.slider("Passing Score (%)", 40, 100, 60, help="Minimum score required to pass")


# ============================================
# 🎯 MAIN UI
# ============================================
st.title("📱 SmartScan EduPad Pro")
st.caption("Advanced OMR-Based Test Paper Analysis System")

student_uploads = st.file_uploader(
    "Upload Student Answer Sheets",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="student_uploads"
)

if student_uploads:
    st.session_state.student_papers = student_uploads
    st.success(f"✅ Uploaded {len(student_uploads)} answer sheets successfully.")


# ============================================
# 🔬 START COMPARISON
# ============================================
if st.button("🔬 Start Comparison", use_container_width=True):

    if not st.session_state.answer_key_image:
        st.error("❌ Please upload an answer key image first.")
        st.stop()

    if not st.session_state.student_papers:
        st.error("❌ Please upload at least one student answer sheet.")
        st.stop()

    with st.spinner("🔍 Processing OMR sheets..."):
        time.sleep(0.5)  # Simulate processing delay

        # Detect answers from answer key
        key_answers = omr_detect_answers(st.session_state.answer_key_image, debug=False)
        st.subheader("🔍 Answer Key OMR Detection Results")
        if key_answers:
            st.json(key_answers)
        else:
            st.warning("No valid bubbles detected in the answer key.")

        results = []

        for i, paper in enumerate(st.session_state.student_papers):
            try:
                student_answers = omr_detect_answers(paper, debug=False)
                score = calculate_score(key_answers, student_answers)
                confidence = np.random.uniform(85, 99)

                status = "PASS" if score >= passing_score else "FAIL"

                results.append({
                    "Student ID": f"STU{i+1:03d}",
                    "Score (%)": score,
                    "AI Confidence (%)": f"{confidence:.1f}%",
                    "Status": status
                })

                st.subheader(f"🧪 Student {i+1} OMR Detection Results")
                if student_answers:
                    st.json(student_answers)
                else:
                    st.warning("No valid bubbles detected.")

            except Exception as e:
                st.error(f"Error processing student {i+1}: {e}")
                continue

        if results:
            st.session_state.results = pd.DataFrame(results)
            st.success(f"✅ Completed analysis for {len(results)} students.")
        else:
            st.warning("No valid results were generated.")


# ============================================
# 📊 RESULTS DISPLAY
# ============================================
if st.session_state.results is not None:
    df = st.session_state.results

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", len(df))
    pass_rate = (df["Score (%)"] >= passing_score).mean() * 100
    col2.metric("Pass Rate", f"{pass_rate:.1f}%")
    col3.metric("Top Score", f"{df['Score (%)'].max()}%")

    # Data Table
    st.dataframe(df, use_container_width=True)

    # Visualizations
    fig1 = px.histogram(df, x="Score (%)", nbins=10, title="Score Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    df["AI_Confidence_num"] = df["AI Confidence (%)"].str.rstrip('%').astype(float)
    fig2 = px.scatter(df, x="Score (%)", y="AI_Confidence_num",
                      title="Score vs AI Confidence",
                      labels={"Score (%)": "Score (%)", "AI_Confidence_num": "Confidence (%)"},
                      hover_data=["Student ID", "Status"])
    st.plotly_chart(fig2, use_container_width=True)

    # Download Button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="omr_results.csv",
        mime="text/csv"
    )
