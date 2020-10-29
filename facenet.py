from keras import backend as K
import time
from multiprocessing.dummy import Pool
K.set_image_data_format('channels_first')
import cv2
import os
import glob
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import win32com.client as wincl

PADDING = 50
ready_to_detect_identity = True
windows10_voice_interface = wincl.Dispatch("SAPI.SpVoice")

FRmodel = faceRecoModel(input_shape=(3, 96, 96))

def triplet_loss(y_true, y_pred, alpha = 0.3):
    """
    Реализация триплетной потери по формуле (3)
    
    Arguments:
    y_pred -- список python, содержащий три объекта:
            anchor -- кодировки для якорных изображений, формы (None, 128)
            positive -- кодировки для положительных образов, формы (None, 128)
            negative -- кодировки для отрицательных изображений, формы (None, 128)
    
    Returns:
    loss -- действительное число, величина потери
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Шаг 1: вычислите (кодирующее) расстояние между anchor и positive, вам нужно будет суммировать по (оси) axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Шаг 2: вычислите (кодирующее) расстояние между anchor и negative, вам нужно будет суммировать по axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Шаг 3: вычтите два предыдущих расстояния и добавьте Альфа
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Шаг 4: возьмите максимум basic_loss и 0.0. Суммируйте по учебным примерам.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
    return loss

FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

def prepare_database():
    database = {}

    # загрузите все изображения людей для распознавания в базу данных
    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = img_path_to_encoding(file, FRmodel)

    return database

def webcam_face_recognizer(database): # Здесь начинаются мутки с вебкамерой (Виктор)
    """
    Запускает цикл, который извлекает изображения с веб-камеры компьютера и определяет, содержит ли
    он лицо человека в нашей базе данных.

    Если он содержит лицо, то будет воспроизведено аудиосообщение, приветствующее пользователя.
    Если нет, то программа обработает следующий кадр с веб-камеры
    """
    global ready_to_detect_identity

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    while vc.isOpened():
        _, frame = vc.read()
        img = frame

        # Мы не хотим обнаруживать новую личность, пока программа находится в процессе идентификации другого человека
        if ready_to_detect_identity:
            img = process_frame(img, frame, face_cascade)   
        
        key = cv2.waitKey(100)
        cv2.imshow("preview", img)

        if key == 27: # выход на ESC
            break
    cv2.destroyWindow("preview")

def process_frame(img, frame, face_cascade):
    """
    Определите, содержит ли текущий кадр лица людей из нашей базы данных
    """
    global ready_to_detect_identity
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Пройдите по всем обнаруженным лицам и определите, есть ли они в базе данных
    identities = []
    for (x, y, w, h) in faces:
        x1 = x-PADDING
        y1 = y-PADDING
        x2 = x+w+PADDING
        y2 = y+h+PADDING

        img = cv2.rectangle(frame,(x1, y1),(x2, y2),(255,0,0),2)

        identity = find_identity(frame, x1, y1, x2, y2)

        if identity is not None:
            identities.append(identity)

    if identities != []:
        cv2.imwrite('example.png',img)

        ready_to_detect_identity = False
        pool = Pool(processes=1) 
        # Мы запускаем это как отдельный процесс, чтобы обратная связь с камерой не замерзала
        pool.apply_async(welcome_users, [identities])
    return img

def find_identity(frame, x1, y1, x2, y2):
    """
    Определите, существует ли грань, содержащаяся в ограничивающем прямоугольнике, в нашей базе данных

    x1,y1_____________
    |                 |
    |                 |
    |_________________x2,y2

    """
    height, width, channels = frame.shape
    # Заполнение необходимо, так как детектор лиц OpenCV создает ограничивающую рамку вокруг лица, а не головы
    part_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    
    return who_is_it(part_image, database, FRmodel)

def who_is_it(image, database, model):
    """
    Реализует распознавание лиц для счастливого дома, находя, кто является человеком на изображении image_path.
    
    Arguments:
    image_path -- путь к изображению
    database -- база данных, содержащая кодировки изображений вместе с именем человека на изображении
    model -- ваш экземпляр начальной модели в Керасе
    
    Returns:
    min_dist -- минимальное расстояние между кодировкой image_path и кодировками из базы данных
    identity -- string, предсказание имени человека на image_path
    """
    encoding = img_to_encoding(image, model)
    
    min_dist = 100
    identity = None
    
    # Цикл над именами и кодировками словаря базы данных.
    for (name, db_enc) in database.items():
        
        # Вычислите расстояние L2 между целевым "encoding" и текущим "emb" из базы данных.
        dist = np.linalg.norm(db_enc - encoding)

        print('distance for %s is %s' %(name, dist))

        # Если это расстояние меньше min_dist, то установите min_dist в dist, а identity - в name
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > 0.52:
        return None
    else:
        return str(identity)

def welcome_users(identities):
    """ Выводит приветственное звуковое сообщение для пользователей """
    global ready_to_detect_identity
    welcome_message = 'Welcome '

    if len(identities) == 1:
        welcome_message += '%s, have a nice day.' % identities[0]
    else:
        for identity_id in range(len(identities)-1):
            welcome_message += '%s, ' % identities[identity_id]
        welcome_message += 'and %s, ' % identities[-1]
        welcome_message += 'have a nice day!'

    windows10_voice_interface.Speak(welcome_message)

    # Разрешить программе снова начать обнаружение идентификационных данных
    ready_to_detect_identity = True

if __name__ == "__main__":
    database = prepare_database()
    webcam_face_recognizer(database)

# ### References:

# 
