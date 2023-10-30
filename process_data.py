import os
import face_detection
import cv2
import torch
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))
video_dir = './data/video'
landmark_dir = './data/landmark'
os.makedirs(landmark_dir, exist_ok=True)
for video in os.listdir(video_dir):
    video_path = os.path.join(video_dir, video)
    video_name = video.split('.')[0]
    landmarks_path = os.path.join(landmark_dir, video.replace('mp4', 'npy'))
    cap = cv2.VideoCapture(video_path)
    landmarks_list = []
    print("Extracting landmarks from {}".format(video_path))
    frame_save_dir = os.path.join('./data/frame', video_name)
    os.makedirs(frame_save_dir, exist_ok=True)
    if len(os.listdir(frame_save_dir)) > 0:
        continue
    count = 0
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break
            frame_save_path = os.path.join(frame_save_dir, f'{count:05d}.jpg')
            cv2.imwrite(frame_save_path, image)
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            count += 1
            landmark_list = []
            if results.multi_face_landmarks is None:
                print(count)
                landmark_list.append(np.zeros((368, 2)))
            else:
                landmarks = results.multi_face_landmarks[0].landmark
                for i in range(len(landmarks)):
                    landmark_list.append(np.array([landmarks[i].x, landmarks[i].y]))
                landmarks_list.append(landmark_list)
    landmarks_numpy = np.array(landmarks_list)
    np.save(landmarks_path, landmarks_numpy)
    cap.release()
mp_face_detection = mp.solutions.face_detection
print('Using {} for inference.'.format(device))
rect_dir = './data/rect'
os.makedirs(rect_dir, exist_ok=True)
for video in os.listdir(video_dir):
    video_path = os.path.join(video_dir, video)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    i = 0
    center_save_path = os.path.join(rect_dir, '{}.npy'.format(video.split('.')[0]))
    print(video, frame_count)
    if os.path.exists(center_save_path):
        rect = np.load(center_save_path)
        if rect.shape[0] == frame_count:
            continue
    print("Detecting faces from {}".format(video_path))
    rect = []
    with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.2) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)
            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detections is None:
                print(i)
                rect.append([0, 0, 0, 0, 0])
            else:
                xmin = results.detections[0].location_data.relative_bounding_box.xmin
                ymin = results.detections[0].location_data.relative_bounding_box.ymin
                width = results.detections[0].location_data.relative_bounding_box.width
                height = results.detections[0].location_data.relative_bounding_box.height
                rect.append([1, xmin, ymin, width, height])
            i += 1
    center = np.array(rect)
    np.save(center_save_path, center)
    cap.release()