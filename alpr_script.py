import cv2
from ultralytics import YOLO
import pytesseract
import time
import numpy as np

# --- Configuration ---
MODEL_PATH = 'C:/Users/WINDOWS/alpr/runs/detect/alpr_yolov5n/weights/best_openvino_model' # Path to your trained YOLOv5 model
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

    results = model(frame, verbose=False, batch=1)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confs, classes):
            if model.names[int(cls)] == 'license_plate' and conf > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box)

                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)

                lp_image = frame[y1:y2, x1:x2]

                # --- OCR Pre-processing ---
                # Convert the cropped image to grayscale first to reduce noise
                gray_lp = cv2.cvtColor(lp_image, cv2.COLOR_BGR2GRAY)

                # Adaptive thresholding
                thresh_lp = cv2.adaptiveThreshold(gray_lp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY_INV, 11, 2)

                # Perform OCR
                lp_text = ""
                try:
                    lp_text = pytesseract.image_to_string(thresh_lp, config=TESSERACT_CONFIG)
                    # Clean up the extracted text: remove non-alphanumeric characters and whitespace
                    lp_text = "".join(filter(str.isalnum, lp_text)).strip().upper()

                    if len(lp_text) > 2:
                        print(f"Detected LP: {lp_text} (Conf: {conf:.2f})")
                        color = (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{lp_text} ({conf:.2f})", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                    else:
                        color = (0, 255, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"LP (No OCR)", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

                except Exception as e:
                    print(f"OCR Error: {e}")
                    color = (0, 0, 255)
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