import cv2
from ultralytics import YOLO

# Load YOLOv8 model (you can change to yolov8s.pt or yolov8x.pt for better accuracy)
model = YOLO("yolov8n.pt")  # Nano = light & fast; use 'yolov8x.pt' for better detection

# Start webcam
cap = cv2.VideoCapture(0)  # Use 0 for default camera

# Check camera status
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame for speed (optional)
    frame = cv2.resize(frame, (640, 480))

    # Run object detection
    results = model(frame)

    # Draw boxes and labels
    annotated_frame = results[0].plot()

    # Show the result
    cv2.imshow("YOLOv8 Live Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()
