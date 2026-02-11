import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tempfile
import os
import time
import uuid

# Local imports
from utils.omr_processor import detect_bubbles
from utils.scorer import calculate_score
from utils.exporter import export_results_csv, export_results_pdf
from utils.pdf_extractor import extract_images_from_pdf

# ============================================
# 🎨 PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="SmartScan EduPad Pro",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set custom CSS for sleek look
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    .stAlert {
        border-left: 5px solid #FF9800;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# 🔧 SESSION STATE INITIALIZATION
# ============================================
if "answer_key_image" not in st.session_state:
    st.session_state.answer_key_image = None
if "student_papers" not in st.session_state:
    st.session_state.student_papers = []
if "results" not in st.session_state:
    st.session_state.results = None
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

# ============================================
# 🔧 SIDEBAR - INPUTS & CONFIG
# ============================================
with st.sidebar:
    st.header("📋 Test Configuration")
    
    # Answer Key Upload
    answer_key_upload = st.file_uploader(
        "Upload Answer Key (Image or PDF)", 
        type=["jpg", "jpeg", "png", "pdf"], 
        accept_multiple_files=False
    )
    
    if answer_key_upload:
        file_ext = answer_key_upload.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
            tmp.write(answer_key_upload.getbuffer())
            temp_path = tmp.name
        
        try:
            if file_ext == "pdf":
                images = extract_images_from_pdf(temp_path)
                if images:
                    img = images[0]  # First page only
                    st.image(img, caption="Answer Key (Extracted)", use_container_width=True)
                else:
                    st.warning("No pages found in PDF.")
            else:
                img = Image.open(temp_path)
                st.image(img, caption="Answer Key", use_container_width=True)
            
            st.session_state.answer_key_image = img
        except Exception as e:
            st.error(f"Error loading answer key: {e}")
        finally:
            os.remove(temp_path)

    # Passing Score
    passing_score = st.slider("Passing Score (%)", 40, 100, 60, help="Minimum score to pass")

    # Debug Toggle
    st.session_state.debug_mode = st.checkbox("Enable Debug Mode", value=False)

    # Reset Button
    if st.button("🔄 Reset Session"):
        st.session_state.clear()
        st.rerun()

# ============================================
# 🎯 MAIN APP LAYOUT
# ============================================
st.title("📱 SmartScan EduPad Pro")
st.caption("AI-Powered OMR Grading System | Supports Images & PDFs")

# Student Paper Upload
st.subheader("📤 Upload Student Answer Sheets")
student_uploads = st.file_uploader(
    "Upload multiple student sheets (JPG, PNG, PDF)", 
    type=["jpg", "jpeg", "png", "pdf"], 
    accept_multiple_files=True
)

if student_uploads:
    st.session_state.student_papers = student_uploads
    st.success(f"✅ Uploaded {len(student_uploads)} papers.")

# ============================================
# 🔬 PROCESSING BUTTON
# ============================================
if st.button("🔬 Start OMR Analysis", type="primary"):
    if not st.session_state.answer_key_image:
        st.error("❌ Please upload an answer key first.")
        st.stop()
    if not st.session_state.student_papers:
        st.error("❌ No student papers uploaded.")
        st.stop()

    with st.spinner("🔍 Analyzing answer sheets..."):
        time.sleep(1)

    # Extract key answers from answer key
    try:
        key_answers = detect_bubbles(st.session_state.answer_key_image, debug=st.session_state.debug_mode)
        st.json({"Answer Key": key_answers})
    except Exception as e:
        st.error(f"Failed to process answer key: {e}")
        st.stop()

    results = []
    progress_bar = st.progress(0)
    total = len(st.session_state.student_papers)

    for i, paper in enumerate(st.session_state.student_papers):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
            tmp.write(paper.getbuffer())
            temp_file = tmp.name

        try:
            if paper.name.lower().endswith(".pdf"):
                images = extract_images_from_pdf(temp_file)
                if not images:
                    st.warning(f"Student {i+1}: No valid pages in PDF.")
                    continue
                img = images[0]
            else:
                img = Image.open(temp_file)

            student_answers = detect_bubbles(img, debug=st.session_state.debug_mode)
            score = calculate_score(key_answers, student_answers)
            status = "PASS" if score >= passing_score else "FAIL"

            results.append({
                "Student ID": f"STU{i+1:03d}",
                "Score (%)": score,
                "Status": status,
                "Answers": student_answers
            })

        except Exception as e:
            st.warning(f"Failed to process student {i+1}: {e}")
        finally:
            os.remove(temp_file)

        progress_bar.progress((i + 1) / total)

    # Store results
    df = pd.DataFrame(results)
    st.session_state.results = df

    # Feedback Prompt
    st.success(f"✅ Completed analysis for {len(results)} students!")
    st.info("💡 Want to improve accuracy? Try adjusting lighting or alignment on next scan.")

# ============================================
# 📊 RESULTS DISPLAY
# ============================================
if st.session_state.results is not None:
    df = st.session_state.results
    
    st.subheader("📊 Final Results")
    st.dataframe(df.style.highlight_max(subset=["Score (%)"]), use_container_width=True)

    # Charts
    fig = px.bar(df, x="Student ID", y="Score (%)", color="Status",
                 color_discrete_map={"PASS": "#4CAF50", "FAIL": "#F44336"},
                 title="Student Scores by Status")
    st.plotly_chart(fig, use_container_width=True)

    # Export Options
    col1, col2 = st.columns(2)
    with col1:
        csv_data = export_results_csv(df)
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name="smartscan_results.csv",
            mime="text/csv"
        )

    with col2:
        pdf_data = export_results_pdf(df)
        st.download_button(
            label="📄 Download as PDF",
            data=pdf_data,
            file_name="smartscan_results.pdf",
            mime="application/pdf"
        )

# ============================================
# 📝 FOOTER
# ============================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:#888; font-size:0.9em'>
        Built with ❤️ for educators • Version 1.2.0 • GitHub Repository: [Click Here](https://github.com/yourusername/smartscan-edupad-pro)
    </div>
    """,
    unsafe_allow_html=True
)
