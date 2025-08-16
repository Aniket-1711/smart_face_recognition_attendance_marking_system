#import opencv for computer vision tasks
# also import os module to organise the folder and paths

import cv2
import os

# load the haar cascade for the face detection (xml must be in the same folder)/
face_cascade=cv2.CascadeClassifier( "C:/Users/gawan/BTECH 2 Year/Python Projects 2025/smart_face_recognition_attendance_marking_system/face_recognition_attendane_ai_system/haarcascade_frontalface_default.xml")

# Initialize the webcam
cap=cv2.VideoCapture(0)

# asking the user to enter id/name for identification purpose further
user_id=input("Enter user ID/Name: ")


# create the dataset folder if it doesn't exist
# if not os.path.exists('dataset'):
#     os.makedirs('dataset')

# initialize a counter to count the no of images captured during the process
count=0

while True:
    ret, frame=cap.read()

    if not ret:
        break

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


    #detect the face
    faces=face_cascade.detectMultiScale(gray,1.1,5)

    # keep cropping and capturing 50 images for all kind of face recognition
    for (x,y,w,h) in faces:
        count+=1

        # crop the face and save it
        face_img=gray[y:y+h,x:x+w]
        file_path="C:/Users/gawan/BTECH 2 Year/Python Projects 2025/smart_face_recognition_attendance_marking_system/face_recognition_attendane_ai_system/dataset/"
        cv2.imwrite(f'{file_path}{user_id}_{count}.jpg',face_img) #takes file path and img to be stored at that path as parameters

        # draw the rectangle around the face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        # display the image count
        cv2.putText(frame,f'Image: {count}', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        #   cv2.putText(img|, text|,           org|,            |font,       fontScale |, color|, |thickness, lineType(optinal))
        # this put text method is used to write content inside the text at specific coordinates, here x,y-10 position with so and so font

    # show the frame
    cv2.imshow('Capturing Faces ',frame)

    if cv2.waitKey(1)==ord('q') or count>=100:
        break
    
cap.release()
cv2.destroyAllWindows()

print(f"/n[Info] collected {count} face samples for {user_id}")

