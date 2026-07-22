import cv2
from inference import get_model
import supervision as sv

API_KEY = "Nz2Z5jwsCQBEkSutNlJS"
MODEL_ID = "asl-ybz8z/2"

# Load model
model = get_model(model_id=MODEL_ID, api_key=API_KEY)

# Supervision annotators
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.5)

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not accessible")

print("Starting webcam...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model.infer(frame)

    # Convert Roboflow response → Supervision detections
    detections = sv.Detections.from_inference(results)

    # Annotate frame
    frame = box_annotator.annotate(frame, detections)
    frame = label_annotator.annotate(frame, detections)

    cv2.imshow("ASL Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
