import fitz  # PyMuPDF
from PIL import Image
import io

def extract_images_from_pdf(pdf_path):
    """Extract all images from PDF and return list of PIL Images."""
    doc = fitz.open(pdf_path)
    images = []

    for page in doc:
        pix = page.get_pixmap(dpi=150)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append(img)

    doc.close()
    return images
