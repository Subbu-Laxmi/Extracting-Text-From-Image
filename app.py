from flask import Flask, render_template, request
import cv2
import pytesseract
import numpy as np
import os

app = Flask(__name__)

# Configure Tesseract OCR - Update the path if necessary
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def process_image(image_path):
    """Preprocess image and extract text using OCR."""
    img = cv2.imread(image_path)

    if img is None:
        return "Error: Unable to read the image!"

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize the image (scaling up improves OCR accuracy)
    scale_percent = 150  # Scale up by 150%
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)

    # Apply bilateral filter (denoising while keeping edges)
    filtered = cv2.bilateralFilter(resized, 9, 75, 75)

    # Sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(filtered, -1, kernel)

    # Apply thresholding
    _, thresh = cv2.threshold(sharpened, 150, 255, cv2.THRESH_BINARY)

    # Extract text using Tesseract OCR
    extracted_text = pytesseract.image_to_string(thresh, lang="eng", config="--psm 6")

    return extracted_text

@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    extracted_text = None
    image_filename = None
    if request.method == "POST":
        if "image" not in request.files:  # Fix: Match the form field name
            return "No file part"
        
        file = request.files["image"] 
        
        if file.filename == "":
            return "No selected file"
        
        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            extracted_text = process_image(file_path)
            image_filename = file.filename
    
    return render_template("upload.html", text=extracted_text, image=image_filename)

if __name__ == "__main__":
    app.run(debug=True)
