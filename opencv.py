import cv2
import time
from ultralytics import YOLO

model_path = r'D:\OPENCV_PROJECT\best.pt'
print("Loading model...")
model = YOLO(model_path)
print("Model loaded.")

video_paths = [
    r'D:\OPENCV_PROJECT\1086212810-preview.mp4'
]

class_mapping = {0: 'sitting', 1: 'standing'}

sitting_threshold = 60  
standing_threshold = 60  
next_person_id = 1
person_trackers = {}
trackers = {}


def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

for video_path in video_paths:
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        continue
    else:
        print(f"Video file {video_path} opened successfully.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        results = model(frame)
        current_boxes = []
        current_trackers = []

        for result in results:
            boxes = result.boxes.xyxy 
            classes = result.boxes.cls  

            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = map(int, box)  
                label = class_mapping.get(int(cls), 'sitting/standing')
                current_boxes.append((x1, y1, x2, y2))

               
                matched_id = None
                highest_iou = 0
                for person_id, data in person_trackers.items():
                    iou = calculate_iou(data['box'], (x1, y1, x2, y2))
                    if iou > 0.5 and iou > highest_iou:  
                        matched_id = person_id
                        highest_iou = iou

                if matched_id is None:
                    matched_id = next_person_id
                    person_trackers[matched_id] = {'box': (x1, y1, x2, y2), 'start_time_sitting': None, 'start_time_standing': None, 'image_saved': False}
                    next_person_id += 1
                    tracker = cv2.TrackerKCF_create()
                    trackers[matched_id] = tracker
                    tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                else:
                    person_trackers[matched_id]['box'] = (x1, y1, x2, y2)

                
                elapsed_time_sitting = 0
                elapsed_time_standing = 0

                
                if label == 'sitting':
                    if person_trackers[matched_id]['start_time_sitting'] is None:
                        person_trackers[matched_id]['start_time_sitting'] = time.time()
                        person_trackers[matched_id]['start_time_standing'] = None 
                    elapsed_time_sitting = time.time() - person_trackers[matched_id]['start_time_sitting']

                
                if label == 'standing':
                    if person_trackers[matched_id]['start_time_standing'] is None:
                        person_trackers[matched_id]['start_time_standing'] = time.time()
                        person_trackers[matched_id]['start_time_sitting'] = None  
                    elapsed_time_standing = time.time() - person_trackers[matched_id]['start_time_standing']

               
                color = (0, 255, 0)  
                if elapsed_time_sitting > sitting_threshold or elapsed_time_standing > standing_threshold:
                    color = (0, 0, 255)  

                
                if (elapsed_time_sitting > sitting_threshold or elapsed_time_standing > standing_threshold) and not person_trackers[matched_id].get('image_saved'):
                    print(f"Capturing cropped image for Person {matched_id}...")
                    cropped_image = frame[y1:y2, x1:x2]  
                    cv2.imwrite(f'D:\\OPENCV_PROJECT\\person_{matched_id}_sitting_or_standing_cropped.jpg', cropped_image)
                    person_trackers[matched_id]['image_saved'] = True  

                
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

              
                font_scale = 0.1  
                thickness = 1  

                if label == 'sitting':
                    frame = cv2.putText(frame, f'Person {matched_id}: Sitting: {int(elapsed_time_sitting)}s', 
                                        (int(x1), int(y2) + 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                        font_scale, color, thickness)
                elif label == 'standing':
                    frame = cv2.putText(frame, f'Person {matched_id}: Standing: {int(elapsed_time_standing)}s', 
                                        (int(x1), int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                                        font_scale, color, thickness)


       
        tracked_ids = list(trackers.keys())
        for person_id in tracked_ids:
            success, box = trackers[person_id].update(frame)
            if success:
                x, y, w, h = [int(i) for i in box]
                person_trackers[person_id]['box'] = (x, y, x + w, y + h)
            else:
                del trackers[person_id]
                del person_trackers[person_id]

        cv2.imshow('Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting current video...")
            break

    cap.release()
    cv2.destroyAllWindows()  

print("All videos processed.")
