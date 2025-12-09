import os
from pathlib import Path
import cv2
from ultralytics import YOLO
import xgboost as xgb
import pandas as pd
import numpy as np
import torch


class Detector:
    def __init__(self, yolo_path=None, xgb_path=None, conf_threshold=0.75, device='auto', imgsz=640):
        base = Path(__file__).resolve().parent
        self.yolo_path = str(yolo_path or base / "best.pt")
        self.xgb_path = str(xgb_path or base / "model_weights.json")
        self.conf_threshold = conf_threshold
        self.imgsz = int(imgsz)

        # determine device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Load models once
        self.model_yolo = YOLO(self.yolo_path)
        try:
            if self.device.startswith('cuda'):
                # move model to GPU
                self.model_yolo.to('cuda:0')
        except Exception:
            pass

        self.model_xgb = xgb.Booster()
        self.model_xgb.load_model(self.xgb_path)

    def predict_frame(self, frame):
        """Run detection on a single BGR OpenCV frame.
        Avoid heavy `plot()` calls; draw directly with OpenCV for speed.
        Returns annotated_frame (BGR), summary dict, and suspicious events list.
        """
        # run inference (ultralytics accepts BGR frames)
        results = self.model_yolo(frame, verbose=False, device=self.device, imgsz=self.imgsz)

        annotated_frame = frame.copy()
        summary = {"suspicious": 0, "normal": 0, "total_boxes": 0}
        suspicious_events = []  # track suspicious detections for alert agent

        for r in results:
            # boxes tensor-like: each row [x1,y1,x2,y2]
            bound_box = r.boxes.xyxy
            confs = []
            try:
                confs = r.boxes.conf.tolist()
            except Exception:
                confs = []

            # some models supply keypoints
            keypoints = []
            try:
                keypoints = r.keypoints.xyn.tolist()
            except Exception:
                keypoints = []

            n_boxes = len(bound_box)
            summary["total_boxes"] += n_boxes

            for index in range(n_boxes):
                try:
                    box = bound_box[index]
                    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                except Exception:
                    continue

                conf_score = confs[index] if index < len(confs) else 0.0

                if conf_score > self.conf_threshold and keypoints:
                    data = {}
                    for j in range(len(keypoints[index])):
                        data[f'x{j}'] = keypoints[index][j][0]
                        data[f'y{j}'] = keypoints[index][j][1]

                    df = pd.DataFrame(data, index=[0])
                    dmatrix = xgb.DMatrix(df)
                    cut = self.model_xgb.predict(dmatrix)
                    pred = int((cut > 0.5).astype(int)[0])

                    if pred == 0:
                        conf_text = f'Suspicious ({conf_score:.2f})'
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 7, 58), 2)
                        cv2.putText(annotated_frame, conf_text, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 7, 58), 2)
                        summary["suspicious"] += 1
                        suspicious_events.append({
                            "confidence": conf_score,
                            "keypoints": keypoints[index],
                            "box": (x1, y1, x2, y2)
                        })
                    else:
                        conf_text = f'Normal ({conf_score:.2f})'
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (57, 255, 20), 2)
                        cv2.putText(annotated_frame, conf_text, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.7, (57, 255, 20), 2)
                        summary["normal"] += 1

        return annotated_frame, summary, suspicious_events


if __name__ == "__main__":
    print("Run `streamlit run app.py` to start the demo UI.")
