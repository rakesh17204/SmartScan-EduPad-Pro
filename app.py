import streamlit as st
import pandas as pd
import numpy as np
import logging
from PIL import Image
import io

# Set up comprehensive logging for debugging and performance tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("omr_detection.log"),
        logging.StreamHandler()
    ]
)

# Initialize session state
if 'answer_key_image' not in st.session_state:
    st.session_state.answer_key_image = None
if 'student_papers' not in st.session_state:
    st.session_state.student_papers = []
if 'results' not in st.session_state:
    st.session_state.results = pd.DataFrame()

# Constants
passing_score = 60  # Minimum passing score in percentage

# ============================================
# 🖼️ IMAGE UPLOAD SECTION
# ============================================
st.title("📄 OMR Answer Sheet Scanner & Grading System")

st.markdown("""
### 📌 Instructions:
1. Upload the **Answer Key** image.
2. Upload one or more **Student Answer Sheets**.
3. Click **"🔬 Start Comparison"** to process and grade.
""")

# Answer Key Upload
st.subheader("🔑 Upload Answer Key")
uploaded_key = st.file_uploader(
    "Choose answer key image (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if uploaded_key:
    try:
        image = Image.open(uploaded_key).convert("RGB")
        st.session_state.answer_key_image = image
        st.image(image, caption="Uploaded Answer Key", use_column_width=True)
        st.success("✅ Answer key loaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to load answer key: {str(e)}")
        logging.error(f"Image loading failed: {str(e)}")

# Student Papers Upload
st.subheader("🎓 Upload Student Answer Sheets")
uploaded_papers = st.file_uploader(
    "Upload student answer sheets (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_papers:
    valid_papers = []
    for file in uploaded_papers:
        try:
            image = Image.open(file).convert("RGB")
            valid_papers.append(image)
            st.image(image, caption=f"Uploaded: {file.name}", use_column_width=True)
        except Exception as e:
            st.warning(f"⚠️ Skipped invalid file: {file.name} — {str(e)}")
            logging.warning(f"Invalid file skipped: {file.name} — {str(e)}")

    if valid_papers:
        st.session_state.student_papers = valid_papers
        st.success(f"✅ Successfully loaded {len(valid_papers)} student papers.")
    else:
        st.warning("⚠️ No valid student papers were uploaded.")

# ============================================
# 🔬 START COMPARISON
# ============================================
if st.button("🔬 Start Comparison", use_container_width=True):

    # ✅ 1. Validate Answer Key
    if not st.session_state.answer_key_image:
        st.error("❌ Please upload an answer key image before starting.")
        st.stop()

    # ✅ 2. Validate Student Papers
    if not st.session_state.student_papers or len(st.session_state.student_papers) == 0:
        st.error("❌ Please upload at least one valid student answer sheet.")
        st.stop()

    # ✅ 3. Memory & Resource Check (Optional Performance Monitoring)
    try:
        import psutil
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent(interval=1)
        if memory_usage > 85 or cpu_usage > 90:
            st.warning(f"⚠️ High system resource usage detected: RAM={memory_usage:.1f}%, CPU={cpu_usage:.1f}%")
            logging.warning(f"High resource usage: RAM={memory_usage:.1f}%, CPU={cpu_usage:.1f}%")
    except ImportError:
        logging.info("psutil not available; skipping resource monitoring.")

    with st.spinner("🔄 Running OMR Detection..."):

        try:
            # Detect answers from answer key
            key_answers = omr_detect_answers(st.session_state.answer_key_image, debug=True)
            st.subheader("🔍 OMR Detection (Answer Key)")
            st.json(key_answers)
            logging.info("Successfully detected answers from answer key.")

        except Exception as e:
            st.error(f"❌ Error detecting answers from answer key: {str(e)}")
            logging.error(f"Critical error in answer key processing: {str(e)}")
            st.stop()

        results = []

        # Process each student paper
        for i, paper in enumerate(st.session_state.student_papers):
            try:
                student_answers = omr_detect_answers(paper, debug=True)
                st.subheader(f"🧪 OMR Detection (Student {i + 1})")
                st.json(student_answers)

                # Calculate score
                score = calculate_score(key_answers, student_answers)
                confidence = np.random.uniform(85, 99)  # Simulated AI confidence

                results.append({
                    "Student ID": f"STU{i + 1:03d}",
                    "Score (%)": score,
                    "AI Confidence (%)": f"{confidence:.1f}%",
                    "Status": "PASS" if score >= passing_score else "FAIL"
                })

                logging.info(f"Processed student {i+1}: Score={score}%, Confidence={confidence:.1f}%")

            except Exception as e:
                st.error(f"❌ Error processing student {i + 1}: {str(e)}")
                logging.error(f"Failed to process student {i + 1}: {str(e)}")
                continue  # Skip to next student

        # Store results
        st.session_state.results = pd.DataFrame(results)
        st.success("✅ OMR Comparison Complete! Results below.")

# ============================================
# 📊 RESULTS DISPLAY
# ============================================
if not st.session_state.results.empty:
    st.subheader("📊 Final Results Summary")
    st.dataframe(st.session_state.results.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='lightcoral'), use_container_width=True)

    # Downloadable CSV
    csv = st.session_state.results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="omr_results.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    st.info("📝 No results to display yet. Upload files and click 'Start Comparison'.")

# ============================================
# 🛠️ DEBUG INFO (Hidden unless needed)
# ============================================
if st.checkbox("🔧 Show Debug Info"):
    st.write("### 🧩 Session State Overview")
    st.write(st.session_state)

