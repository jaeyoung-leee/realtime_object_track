import cv2
import base64
import numpy as np
import json
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class ModelHandler:
    def __init__(self):
        self.model = YOLO("/Users/two_jyy/project_openSW/thinkthing/yolov10n.pt")  # 모델 경로를 변경
        self.model.conf = 0.8  # Confidence threshold 설정
        self.tracker = DeepSort(max_age=30, n_init=3)  # DeepSort 초기화
        
    def process_image(self, encoded_data):
        try:
            encoded_data = encoded_data.split(',')[1] if ',' in encoded_data else encoded_data
            frame = base64.b64decode(encoded_data)
            nparr = np.frombuffer(frame, np.uint8)
        except Exception as e:
            return json.dumps({"error": f"Error decoding base64 image data: {str(e)}"}), f"Error decoding base64 image data: {str(e)}"

        if nparr.size == 0:
            return json.dumps({"error": "No data in buffer"}), "No data in buffer"

        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return json.dumps({"error": "Failed to decode the image"}), "Failed to decode the image"

        try:
            results = self.model.track(img, persist=True)
        except Exception as e:
            return json.dumps({"error": f"Error during model inference: {str(e)}"}), f"Error during model inference: {str(e)}"

        if not results or len(results) == 0 or not results[0].boxes:
            return json.dumps({"error": "No detections"}), "No detections"

        detections = []
        try:
            for result in results:
                for obj in result.boxes:
                    bbox = obj.xyxy[0].cpu()
                    conf = float(obj.conf[0].cpu().numpy())
                    class_id = obj.cls.int().cpu().tolist()
                
                    box_dict = {
                        "coordinates": bbox,
                        "confidence": conf,
                        "class": class_id,
                    }
                    detections.append(box_dict.copy())

            print("Detections before tracking:", detections)

            tracks = self.tracker.update_tracks(detections, frame=img)
            print("Tracks:", tracks)

            for track in tracks:
                if track.is_confirmed() and track.time_since_update <= 1:
                    bbox = track.to_tlbr()
                    track_id = track.track_id
                    
                    box_dict={
                        "coordinates" : [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                        "track_id" : track_id
                    }
                    detections[-1].update(box_dict)

            print("Detections after tracking:", detections)

        except Exception as e:
            return json.dumps({"error": f"Error processing results: {str(e)}"}), f"Error processing results: {str(e)}"

        print("Final detections:", detections)
        return json.dumps(detections), None
