import os
import cv2
import time


import cv2
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_single_image(img_path, face_cascade, save_path):
    if not os.path.isfile(img_path):
        return
    
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return

    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    faces = face_cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img_color, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if save_path is not None:
        output_path = os.path.join(save_path, os.path.splitext(os.path.basename(img_path))[0] + ".jpg")
        cv2.imwrite(output_path, img_color)

def face_detect(img_paths, save_path="./output/Haar_output", max_workers=16):
    # use Haar Cascade face detector
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_image, img_path, face_cascade, save_path) for img_path in img_paths]


if __name__ == "__main__":
    img_paths = os.listdir("./dataset/BioID-FaceDatabase")
    img_paths = [os.path.join("./dataset/BioID-FaceDatabase", name) for name in img_paths]
    start = time.time()
    face_detect(img_paths)
    end = time.time()
    execute_time = end - start
    fps = len(img_paths) / execute_time

    print("execution time is", execute_time)
    print("frame per second is", fps)
