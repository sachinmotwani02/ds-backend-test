from flask import Flask, render_template, request, send_file
import os
import cv2
import easyocr
from datetime import datetime
from ultralytics import YOLO

model = YOLO('yolo12s.pt',task='detect')
logo = cv2.imread("defect-scanner-logo-transparent-cropped.png")
model.predict(
        logo, conf=0.6, iou=0.4, imgsz=640, verbose=False, device ='cpu')
# Initialize EasyOCR Reader (disable GPU for CPU mode)
reader = easyocr.Reader(['en'], gpu=False)

app = Flask(__name__)
UPLOAD_FOLDER = 'static/clicked_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/log_delta', methods=['POST'])
def log_delta():
    data = request.get_json()
    print("Delta from browser:", data.get('delta'))
    return '', 204  # No Content

@app.route('/upload', methods=['POST'])
def upload():
    global model, reader
    file = request.files['image']
    gallery_count = request.form.get('gallery_count', type=int)

    if not file:
        return 'No file uploaded', 400

    # Save uploaded image
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Load with OpenCV
    image = cv2.imread(filepath)
    h, w, _ = image.shape

    print("now detect OCR: ",gallery_count)
    if gallery_count!=3:
        # yolo detection
        results = model.predict(
            image, conf=0.6, iou=0.4, imgsz=640, verbose=False, device ='cpu')
        image = results[0].plot( labels=True,conf=False)
        # # Draw a rectangle in the center
        # center_x, center_y = w // 2, h // 2
        # box_width, box_height = w // 4, h // 4
        # top_left = (center_x - box_width // 2, center_y - box_height // 2)
        # bottom_right = (center_x + box_width // 2, center_y + box_height // 2)
        # cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 3)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # OCR using EasyOCR
        results = reader.readtext(rgb)
        # Draw results on the image
        for (bbox, text, prob) in results:
            # Extract coordinates
            (x, y, w, h) = bbox[0][0], bbox[0][1], bbox[2][0] - bbox[0][0], bbox[2][1] - bbox[0][1]
            
            # Draw bounding box and text
            cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            cv2.putText(image, text, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


    # Save modified image
    modified_filename = f"modified_{filename}"
    modified_filepath = os.path.join(UPLOAD_FOLDER, modified_filename)
    cv2.imwrite(modified_filepath, image)

    return {'image_url': f"/{modified_filepath}"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, 
            ssl_context=(r'C:\project2\SE_singapore\ssl\certificaten.crt',
             r'C:\project2\SE_singapore\ssl\private.key'), debug=False)
