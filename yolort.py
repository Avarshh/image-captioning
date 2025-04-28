import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt") 
cap = cv2.VideoCapture(0)  


if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

   
    frame = cv2.resize(frame, (640, 480))

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Live Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()
