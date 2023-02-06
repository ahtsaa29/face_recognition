# It captures images and stores them in datasets resp folder under the folder name of sub_data
import cv2, os
# catch face classics
haar_file = 'haarcascade_frontalface_default.xml'
face_cap = cv2.CascadeClassifier(haar_file)
datasets = 'datasets'  
video_cap = cv2.VideoCapture(0)

def face_store():
# These are sub data sets of folder for faces 
    sub_data = input('Enter Your Name:').capitalize()

    path = os.path.join(datasets, sub_data)
    if not os.path.isdir(path):
        os.mkdir(path)
    
    # defining the size of images 
    (width, height) = (130, 100)    

    
    # The program loops until it has 30 images of the face.
    count = 0
    while count < 30: 
        (_, cap_data) = video_cap.read()
        col = cv2.cvtColor(cap_data, cv2.COLOR_BGR2GRAY)
        faces = face_cap.detectMultiScale(col, 1.3, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(cap_data, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = col[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            cv2.imwrite('% s/% s.png' % (path, count), face_resize)
        count += 1
        
        cv2.imshow('Reading your image', cap_data)
        if cv2.waitKey(100) == ord("z"):
            break

face_store()