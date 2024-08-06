from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

model = YOLO('/Users/two_jyy/project_openSW/model/yolov10n.pt')

tracker = DeepSort(max_age=30, n_init=3)

# 웹캠 열기
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 의미합니다. 다른 웹캠을 사용할 경우 인덱스를 변경하세요.

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8로 객체 탐지
    results = model(frame)

    # 감지된 객체 정보 가져오기
    detections = []
    for result in results:
        for obj in result.boxes:
            bbox = obj.xyxy[0].cpu().numpy()  # 바운딩 박스 좌표
            conf = obj.conf[0].cpu().numpy()  # 신뢰도
            class_id = obj.cls[0].cpu().numpy()  # 클래스 ID

            # 클래스 ID가 "person" (사람)을 나타내는 경우를 필터링
            if class_id != 0:  # COCO 데이터셋 기준으로 "person"의 클래스 ID는 0
                detections.append((bbox, conf, class_id))

    # Deep SORT로 객체 추적
    tracks = tracker.update_tracks(detections, frame=frame)

    # 추적된 객체 그리기
    for track in tracks:
        if track.is_confirmed() and track.time_since_update <= 1:
            bbox = track.to_tlbr()
            track_id = track.track_id
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # 결과 프레임 출력
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
