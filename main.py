import cv2
import supervision as sv
from inference.models.utils import get_roboflow_model
from threading import Thread
import time
import numpy as np

# --- CONFIGURATION ---
API_KEY = "Nz2Z5jwsCQBEkSutNlJS"
MODEL_ID = "asl-ybz8z/2"
CONFIDENCE = 0.4  # Slightly lower helps with CPU speed
# Inference size (what the model sees)
INF_SIZE = (320, 240) 
# Original size (what you see)
SRC_SIZE = (640, 480)

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, SRC_SIZE[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, SRC_SIZE[1])
        # Force a small buffer to reduce lag
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        t = Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if grabbed:
                self.frame = frame

    def stop(self):
        self.stopped = True
        self.stream.release()

def main():
    
    model = get_roboflow_model(model_id=MODEL_ID, api_key=API_KEY)

    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    prev_time = 0

    while True:
        frame = vs.frame
        if frame is None: continue

        # 1. Resize for the model
        inf_frame = cv2.resize(frame, INF_SIZE)

        # 2. Run Inference
        results = model.infer(inf_frame, confidence=CONFIDENCE)[0]
        
        # 3. FIX: Rescale detections back to original size (640x480)
        detections = sv.Detections.from_inference(results)
        
        # This multiplier maps 320x240 back to 640x480
        # Formula: original / resized
        multiplier = np.array([
            SRC_SIZE[0]/INF_SIZE[0], 
            SRC_SIZE[1]/INF_SIZE[1], 
            SRC_SIZE[0]/INF_SIZE[0], 
            SRC_SIZE[1]/INF_SIZE[1]
        ])
        detections.xyxy = detections.xyxy * multiplier

        # 4. Annotate the ORIGINAL high-res frame
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        # FPS Calc
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Optimized Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()