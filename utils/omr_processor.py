import cv2
import numpy as np
import pytesseract
from PIL import Image

def detect_bubbles(image, debug=False):
    """
    Detect filled bubbles using contour analysis.
    Returns dictionary of question -> answer choice.
    """
    if isinstance(image, Image.Image):
        img = np.array(image.convert('RGB'))
    else:
        img = image.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter small contours
    min_area = 10
    max_area = 150
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h
            if 0.8 < aspect_ratio < 1.2:
                candidates.append((x, y, w, h))

    # Sort by row and column (top-to-bottom, left-to-right)
    candidates.sort(key=lambda c: (c[1], c[0]))

    # Group into rows and columns
    questions = {}
    current_row = -1
    current_col = -1
    row_threshold = 50
    col_threshold = 50

    for idx, (x, y, w, h) in enumerate(candidates):
        row = y // row_threshold
        col = x // col_threshold

        if abs(row - current_row) > 1 or abs(col - current_col) > 1:
            # New question
            q_num = len(questions) + 1
            questions[q_num] = 'A'  # Default to A

        # Assign answer based on position
        if col % 5 == 0:
            questions[len(questions) + 1] = 'A'
        elif col % 5 == 1:
            questions[len(questions) + 1] = 'B'
        elif col % 5 == 2:
            questions[len(questions) + 1] = 'C'
        elif col % 5 == 3:
            questions[len(questions) + 1] = 'D'
        elif col % 5 == 4:
            questions[len(questions) + 1] = 'E'

        current_row = row
        current_col = col

    # Optional: show debug overlay
    if debug and len(candidates) > 0:
        debug_img = img.copy()
        for x, y, w, h in candidates:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        st.image(debug_img, caption="Debug: Detected Bubbles", use_column_width=True)

    return questions
