import pandas as pd
import numpy as np
import cv2
from ultralytics import YOLO

def get_line_side(x, y, line_start, line_end): 
    return np.sign((line_end[0] - line_start[0])*(y - line_start[1]) - 
                   (line_end[1] - line_start[1])*(x - line_start[0]))

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("IMG_5268.MOV")

if not cap.isOpened():
    exit("Video açılamadı")

success, frame = cap.read()
if not success or frame is None:
    exit("Video açılmadı veya ilk kare okunamadı")

frame_height, frame_width = frame.shape[:2]

# İlk hesaplama (doğru (x,y) sırası: x = width, y = height)
line_start = (int(frame_width * 0.5), frame_height)
line_end = (frame_width, int(frame_height * 0.2))

while True:
    success, frame = cap.read()
    if not success or frame is None:
        print("Video bitti veya kare okunamadı.")
        break

    frame = cv2.resize(frame, None, fx=0.6, fy=0.6)

    results = model.track(frame, persist=True, stream=False, conf = 0.5, iou = 0.5, tracker = "bytetrack.yaml")

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.int().tolist()
        classes = results[0].boxes.cls.int().tolist()
        xyxy = results[0].boxes.xyxy

        for i, box in enumerate(xyxy):
            cls_id = classes[i]
            track_id = ids[i]
            class_name = model.names[cls_id]
            if class_name not in counts:
                continue
            x1,y1,x2,y2 = map(int, box)
            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)

            current_side = get_line_side(cx, cy, line_start, line_end)
            previous_side = object_last_side.get(track_id, None)
            object_last_side[track_id] = current_side

            if previous_side is not None and previous_side != current_side:
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                    counts[class_name] += 1

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{class_name} ID: {track_id}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.circle(frame, (cx,cy), 4, (255,0,0), 2)


            
    frame_height, frame_width = frame.shape[:2]
    line_start = (int(frame_width * 0.1), int(frame_height))
    line_end   = (int(frame_width * 0.9), int(frame_height * 0.2))

    cv2.line(frame, line_start, line_end, (0,0,255), 2)
    y_offset = 30
    for cls, count in counts.items():
        text = f"{cls}: {count}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30 
    
    
    cv2.imshow("araç takip ve sayım", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): 
        break

cap.release()
cv2.destroyAllWindows()

counts = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0}
counted_ids = set()
object_last_side = {}


