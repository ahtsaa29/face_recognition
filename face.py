import cv2


# catch face classics
face_cap = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# for opening run time camera
video_cap = cv2.VideoCapture(0)
while True:
    # read
    ret, video_data = video_cap.read()
    # black and white image
    col = cv2.cvtColor(video_data,cv2.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor= 1.1,
        minNeighbors =5,
        minSize = (30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    for (x,y,w,h ) in faces:
        cv2.rectangle(video_data,(x,y),(x+w,y+h),(255,255,0), 2)

    # frame
    cv2.imshow("Video_Live",video_data)
    # to stop video when a
    if cv2.waitKey(10) == ord("z"):
        break

video_cap.release()

