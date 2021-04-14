import cv2
import dlib
from FaceDetection.FaceManager import FaceDetectorSSD

cam = cv2.VideoCapture(0)
fd = FaceDetectorSSD()
print("Pretending to feeding this to Neural Net:")
while True:
    ret_val, img = cam.read()
    result = fd.detect_face(img)

    indices = [0, 0, 0, 0] if not result else result["bbox_indices"]
    top, bottom, left, right = indices
    rect = dlib.rectangle(left, top, right, bottom)
    landmarks = fd.get_face_landmarks(img, rect)
    aligned_and_detected = fd.align(img, landmarks)
    print(aligned_and_detected[:50])
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow('RealTimeTest', img)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
