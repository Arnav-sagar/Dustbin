import cv2
from ultralytics import YOLO

# Load model (use 'yolo11n.pt' for testing, or your custom 'best.pt' trained on your trash classes)
model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define the exact class names or IDs you want to track
# If using COCO pre-trained model: 'bottle' (39), 'cup' (41)
# If using a custom model, match these to your custom class labels (e.g., ['paper_crumble', 'bottle', 'paper_cup'])
TARGET_CLASSES = ['bottle', 'cup', 'paper_crumble'] 

print("--- Filtering for Trash Only. Press 'q' to quit. ---")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run ByteTrack inference
    results = model.track(
        frame, 
        persist=True, 
        tracker="bytetrack.yaml", 
        conf=0.4, 
        verbose=False
    )

    annotated_frame = frame.copy()

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()  # Center X, Y, Width, Height
        track_ids = results[0].boxes.id.int().cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()  # Class indices detected
        confidences = results[0].boxes.conf.cpu().numpy()

        for box, track_id, cls_id, conf in zip(boxes, track_ids, classes, confidences):
            class_name = model.names[int(cls_id)]
            
            # Filter: Only process if the detected object is in our allowed trash list
            if class_name in TARGET_CLASSES:
                x_center, y_center, width, height = box
                
                # Draw custom bounding box and label for filtered trash
                x1, y1 = int(x_center - width/2), int(y_center - height/2)
                x2, y2 = int(x_center + width/2), int(y_center + height/2)
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"TRASH: {class_name} ID:{track_id} ({conf:.2f})"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Action trigger coordinates for your launcher/bin mechanism
                print(f"Target Acquired -> [{class_name}] ID:{track_id} at X:{int(x_center)}, Y:{int(y_center)}")

    cv2.imshow("Trash-Only Airborne Tracker", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()