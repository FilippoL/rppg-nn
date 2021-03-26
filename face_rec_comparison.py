import cv2
import face_recognition
from tqdm import tqdm

input_movie = cv2.VideoCapture("H:\\DatasetsThesis\\ECG-FITNESS\\00\\01\\c920-1.avi")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('output.avi', fourcc, 30, (1920, 1080))

face_locations = []
face_encodings = []
face_names = []
frame_number = 0

file = open("H:\\DatasetsThesis\\ECG-FITNESS\\bbox\\00\\01\\c920-1.face", mode='r')
whole_file = file.readlines()
file.close()

for f in tqdm(range(int(length))):
    ret, frame = input_movie.read()
    if not ret:
        break
    rgb_frame = frame[:, :, ::-1]

    new_face_locations = face_recognition.face_locations(rgb_frame)
    (top, right, bottom, left) = new_face_locations[0]
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, "NEW", (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    old_face_locations = [int(float(n)) for n in whole_file[f].split(" ")[1:]]
    (left, top, right, bottom) = old_face_locations
    cv2.rectangle(frame, (left, top), (left + right, top + bottom), (0, 255, 0), 2)
    cv2.rectangle(frame, (left, (top + bottom) - 25), (left + right, top + bottom), (0, 255, 0), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, "OLD", (left + 6, (top + bottom) - 6), font, 0.5, (255, 255, 255), 1)

    output_movie.write(frame)

input_movie.release()
cv2.destroyAllWindows()
