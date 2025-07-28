import cv2
from ultralytics import YOLO
import pytesseract
import time
import numpy as np


#Aron Joshua Holgado
# --- Configuration ---
MODEL_PATH = 'models/best.pt' # Path to your trained YOLOv5 model
CONFIDENCE_THRESHOLD = 0.5            # Minimum confidence to consider a detection
CAMERA_INDEX = 0                      # 0 for default USB webcam, adjust for others (e.g., 1 for another USB, or PiCam might be 0)
# Tesseract configuration for alphanumeric characters
TESSERACT_CONFIG = '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
OUTPUT_WIDTH = 640 # Rescale frame for display, can be same as input for detection
OUTPUT_HEIGHT = 480

# --- Load YOLOv5 Model ---
try:
    model = YOLO(MODEL_PATH)
    print(f"YOLOv5 model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading YOLOv5 model: {e}")
    print("Please ensure the model path is correct and the file exists.")
    exit()

# --- Initialize Camera ---
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Error: Could not open camera with index {CAMERA_INDEX}.")
    print("Please check if the camera is connected and enabled.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, OUTPUT_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, OUTPUT_HEIGHT)

print(f"Camera initialized at resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

# --- Main Loop ---
fps_start_time = time.time()
fps_frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Optional: Resize frame for faster inference if your model takes smaller input
    # Note: YOLOv5 handles resizing internally, but you can explicitly resize
    # if you want to control the input resolution for detection.
    # resized_frame = cv2.resize(frame, (640, 640)) # Example for 640x640 input

    # Perform inference
    # If your model was trained on 640x640, model(frame) will handle scaling.
    # You can specify imgsz= to force a specific size for inference.
    results = model(frame, verbose=False) # verbose=False suppresses prediction messages

    # Process results
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()  # Bounding box coordinates (x1, y1, x2, y2)
        confs = r.boxes.conf.cpu().numpy()  # Confidence scores
        classes = r.boxes.cls.cpu().numpy() # Class IDs

        for box, conf, cls in zip(boxes, confs, classes):
            # Assuming 'license_plate' is class 0 in your model's names list
            # Check if the detected class is 'license_plate' and confidence is above threshold
            if model.names[int(cls)] == 'license_plate' and conf > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box)

                # Ensure coordinates are within frame boundaries
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)

                # Crop the license plate region
                lp_image = frame[y1:y2, x1:x2]

                # --- OCR Pre-processing ---
                # Convert to grayscale
                gray_lp = cv2.cvtColor(lp_image, cv2.COLOR_BGR2GRAY)

                # Apply adaptive thresholding or Otsu's thresholding for better text extraction
                # Adjust parameters as needed for your lighting conditions
                # _, thresh_lp = cv2.threshold(gray_lp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # Use adaptive thresholding for varying light conditions
                thresh_lp = cv2.adaptiveThreshold(gray_lp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY, 11, 2)

                # Optional: Denoising (useful if plates are grainy)
                # denoised_lp = cv2.fastNlMeansDenoising(gray_lp, None, 10, 7, 21)

                # Optional: Sharpening (if text is blurry)
                # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                # sharpened_lp = cv2.filter2D(gray_lp, -1, kernel)

                # Perform OCR
                lp_text = ""
                try:
                    lp_text = pytesseract.image_to_string(thresh_lp, config=TESSERACT_CONFIG)
                    # Clean up the extracted text: remove non-alphanumeric characters and whitespace
                    lp_text = "".join(filter(str.isalnum, lp_text)).strip().upper()

                    if len(lp_text) > 2: # Only display if some text was found
                        print(f"Detected LP: {lp_text} (Conf: {conf:.2f})")
                        # Draw bounding box and text
                        color = (0, 255, 0) # Green for detected
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{lp_text} ({conf:.2f})", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                    else:
                        # Draw yellow if detection but no meaningful OCR
                        color = (0, 255, 255) # Yellow
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"LP (No OCR)", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

                except Exception as e:
                    print(f"OCR Error: {e}")
                    # Draw red if OCR entirely fails
                    color = (0, 0, 255) # Red
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, "OCR Failed!", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


    # Calculate and display FPS
    fps_frame_count += 1
    if time.time() - fps_start_time >= 1:
        fps = fps_frame_count / (time.time() - fps_start_time)
        print(f"FPS: {fps:.2f}")
        fps_frame_count = 0
        fps_start_time = time.time()

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    # Display the frame
    cv2.imshow('YOLOv5 License Plate Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()