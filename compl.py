# Импортируем все нужные модули
import cv2
from PIL import Image, ImageDraw
import face_recognition
import numpy

# Создаем переменную с алгоритмом Хаара, который мы взяли в открытой базе
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

# Загружаем "знакомые" лица
image1 = face_recognition.load_image_file('timka.jpg')
face_encoding1 = face_recognition.face_encodings(image1)[0]

# Знакомые лица в изображении
known_face_encodings = [
    face_encoding1
]

# Список знакомых лиц
known_face_names = [
    "timka"
]

# Запускаем бесконечный цикл
while True:
    _, frame = video_capture.read()
    # Переводим изображение в серое, чтобы было удобнее с ним работать
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Находим лицо
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Рисуем прямоугольник вокруг каждого лица
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    rgb_frame = frame[:,:,::-1]

    # Еще раз находим лица, но уже для алгоритма распознавания
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (t, r, b, l), face_encodings in zip(face_locations, face_encodings):
        # Сравниваем лицо на видео с лицами в базе  
        matches = face_recognition.compare_faces(known_face_encodings, face_encodings)
        name = 'Unknown'
        # Если лицо есть в базе, то находим его индекс и запрашиваем имя из списка имен
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Выбираем шрифт
        font = cv2.FONT_HERSHEY_DUPLEX
        # Отображаем текс с именем, если лицо есть в базе и без него, если нет
        cv2.putText(frame, name, (l+6, b-6), font, 1.0, (255,255,255), 1)
    # Включаем отображение всего кода на видео
    cv2.imshow('Video', frame)

    # Нажимаем q, удерживаем и выходим из программы
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()