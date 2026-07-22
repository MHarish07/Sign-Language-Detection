import time
import cv2
import supervision as sv
from inference import get_roboflow_model
import numpy as np

class GestureProcessor:
    def __init__(self, model_id, api_key, confidence=0.4, stability_threshold=1.5):
        self.model = get_roboflow_model(model_id=model_id, api_key=api_key)
        self.confidence = confidence
        self.stability_threshold = stability_threshold
        
        # Stability tracking
        self.gesture_start_times = {} # gesture_name -> first_seen_timestamp

    def process_frame(self, frame):
        # 1. Resize for the model
        inf_frame = cv2.resize(frame, (320, 240))

        # 2. Run Inference using the actual self.confidence parameter
        results = self.model.infer(inf_frame, confidence=self.confidence)[0]
        
        # 3. Use Supervision to safely parse Roboflow's results
        detections = sv.Detections.from_inference(results)
        
        raw_detections = []
        for i in range(len(detections)):
            # Supervision bbox format is [x_min, y_min, x_max, y_max]
            bbox = detections.xyxy[i].tolist()
            class_name = detections.data['class_name'][i]
            conf = detections.confidence[i]
            
            raw_detections.append({
                "bbox": bbox,
                "class_name": str(class_name),
                "confidence": float(conf)
            })
        
        # Sort by confidence descending
        raw_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        now = time.time()
        # Track top 3 unique gestures from the detections
        top_gestures = []
        final_detections = [] 
        seen = set()
        
        for det in raw_detections:
            g = det['class_name']
            if g not in seen:
                top_gestures.append(g)
                final_detections.append(det)
                seen.add(g)
            if len(top_gestures) >= 3:
                break
        
        # Update start times
        new_start_times = {}
        for g in top_gestures:
            new_start_times[g] = self.gesture_start_times.get(g, now)
        self.gesture_start_times = new_start_times

        commit_triggered = False
        progress = 0.0
        best_gesture = None
        max_duration = 0

        # Check durations of gestures in top 3
        for g, start_time in self.gesture_start_times.items():
            duration = now - start_time
            if duration > max_duration:
                max_duration = duration
                best_gesture = g
            
            if duration >= self.stability_threshold:
                commit_triggered = True
                best_gesture = g # This one wins
                break 
        
        if commit_triggered:
            self.gesture_start_times = {} # Reset all for next character
            progress = 1.0
        elif best_gesture:
            progress = min(max_duration / self.stability_threshold, 1.0)
        
        # For display, use the one with most progress or the top detection
        display_gesture = best_gesture if best_gesture else (top_gestures[0] if top_gestures else None)
            
        return final_detections, display_gesture, commit_triggered, progress

    def reset_stability(self):
        self.gesture_start_times = {}