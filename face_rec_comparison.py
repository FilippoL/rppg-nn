import cv2
import face_recognition
from tqdm import tqdm
from FaceDetection.FaceFinder import FaceDetectorSSD

fd = FaceDetectorSSD()

input_movie = cv2.VideoCapture("H:\\DatasetsThesis\\ECG-FITNESS\\00\\01\\c920-1.avi")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
resolution = (int(input_movie.get(3)), int(input_movie.get(4)))
print(resolution)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('./data/data_out/ecgfitness_ssd_vs_hog.avi', fourcc, 30, resolution)

face_locations = []
face_encodings = []
face_names = []
frame_number = 0


for f in tqdm(range(length)):
    ret, frame = input_movie.read()
    if not ret:
        break
    rgb_frame = frame[:, :, ::-1]

    new_face_locations = face_recognition.face_locations(rgb_frame)
    (top, right, bottom, left) = new_face_locations[0]

    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, "HOG", (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    face_img, old_face_locations = fd.detect_face(rgb_frame, return_indices=True)
    top, bottom, left, right = old_face_locations

    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.rectangle(frame, (left, bottom + 25), (right, bottom), (0, 255, 0), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, "SSD", (left + 6, bottom + 20), font, 0.5, (255, 255, 255), 1)

    output_movie.write(frame)

input_movie.release()
cv2.destroyAllWindows()
