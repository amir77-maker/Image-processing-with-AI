from ultralytics import YOLO
import cv2
import torch
import numpy as np
import threading
from facenet_pytorch import MTCNN

# Initialize models
detector = YOLO('yolov8n.pt')  # Using a lightweight YOLO variant
compute_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {compute_device}")
face_detector = MTCNN(keep_all=True, device=compute_device)

# Setup webcam stream
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Parameters for distance estimation
REAL_WIDTH = 0.5  # Actual object width in meters
CAMERA_FOCAL = 700  # Focal length in millimeters

def estimate_distance(pixel_width):
    """
    Estimate distance from camera based on object's pixel width.
    """
    if pixel_width > 0:
        return (REAL_WIDTH * CAMERA_FOCAL) / pixel_width
    return None

# Shared resources
current_frame = None
yolo_detections = []
face_detections = []
frame_lock = threading.Lock()

# Thread function: Object Detection
def yolo_thread():
    global current_frame, yolo_detections
    while True:
        with frame_lock:
            if current_frame is None:
                continue
            img = current_frame.copy()

        results = detector(img, conf=0.3)
        detections = []
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = b.conf[0]
                cls_name = r.names[int(b.cls[0])]
                detections.append((x1, y1, x2, y2, conf, cls_name))

        with frame_lock:
            yolo_detections = detections

# Thread function: Face Detection
def face_thread():
    global current_frame, face_detections
    while True:
        with frame_lock:
            if current_frame is None:
                continue
            rgb_img = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

        boxes, _ = face_detector.detect(rgb_img)
        faces = []
        if boxes is not None:
            for b in boxes:
                faces.append(tuple(map(int, b)))

        with frame_lock:
            face_detections = faces

# Launch threads
threading.Thread(target=yolo_thread, daemon=True).start()
threading.Thread(target=face_thread, daemon=True).start()

# Main loop
while True:
    ret, frame_feed = camera.read()
    if not ret:
        print("Error: Frame capture failed.")
        break

    with frame_lock:
        current_frame = frame_feed.copy()

    # Draw YOLO results
    with frame_lock:
        for (x1, y1, x2, y2, conf, cls) in yolo_detections:
            width_px = x2 - x1
            dist = estimate_distance(width_px)

            # Box color logic
            if cls == "person":
                color = (0, 255, 0)
            elif cls == "cell phone":
                color = (0, 0, 255)
            else:
                color = (0, 255, 255)

            cv2.rectangle(frame_feed, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_feed, f'{cls} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if dist is not None:
                cv2.putText(frame_feed, f'Distance: {dist:.2f}m', (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                if cls == "person" and dist > 2.0:
                    cv2.putText(frame_feed, "So far!", (x1, y1 - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Draw face results
    with frame_lock:
        for (fx1, fy1, fx2, fy2) in face_detections:
            cv2.rectangle(frame_feed, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
            cv2.putText(frame_feed, "Face", (fx1, fy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show output window
    cv2.imshow('Live Detection and Distance Estimation', frame_feed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
