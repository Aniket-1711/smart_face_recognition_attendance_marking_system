# import required modules

import cv2
from datetime import datetime   # for getting current date and time
import numpy as np
import os
from PIL import Image


# 1.Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(
    "C:/Users/gawan/BTECH 2 Year/Python Projects 2025/smart_face_recognition_attendance_marking_system/face_recognition_attendane_ai_system/haarcascade_frontalface_default.xml"
)

# 2. Load the LBPH Face Recognizer and trained model
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read( "C:/Users/gawan/BTECH 2 Year/Python Projects 2025/smart_face_recognition_attendance_marking_system/face_recognition_attendane_ai_system/trainer.yml")

# 3 Load the name-ID mapping 
# create a dictionary to store those name -Id 
labels={}
with open("C:/Users/gawan/BTECH 2 Year/Python Projects 2025/smart_face_recognition_attendance_marking_system/face_recognition_attendane_ai_system/name_labels.txt","r") as f:
    for line in f:  #iterate over each line in name_labels.txt
        key,val=line.strip().split(":") #remove trailing and ending spaces and then split by :
        labels[int(key)]=val


# function to mark attendance as soon as face is identified, the name will be sent as parameter to this funcion
# *******------note this function is applicable once per a day, as we r only checking date but not time, by modifying the if condition for checking time we can schedule this attendance for multiple times for each date --------********

filename='attendance.csv'
filepath=f"C:/Users/gawan/BTECH 2 Year/Python Projects 2025/smart_face_recognition_attendance_marking_system/face_recognition_attendane_ai_system/{filename}"


def mark_attendance(name):
    
    now=datetime.now()            #gets the current date and time
    date=now.strftime('%Y-%m-%d') #format date like 2025-05-22
    time=now.strftime('%H:%M:%S') #format time like : 14:30:05


    if os.path.exists(filepath):
        with open(filepath,'r+') as f:
            # read the lines, this will move the pointer to the end of the file
            lines=f.readlines()

            # skip header, check if name and date already exist together
            entries=[line.strip().split(',') for line in lines[1:]]
            # 1: because skipping the header line, which is name, date and time

            # check if this name with today's date is already present
            already_marked = any(entry[0].strip() == name and entry[1].strip() == date for entry in entries)

            if not already_marked:
                f.write(f'{name},{date}, {time}\n')

    # firstly as the .csv file doesn't exist, else will execute
    
    else:
        with open(filepath,'w') as f:
            f.write('Name, Date, Time\n') #write header if file doesn't exist
            f.write(f'{name}, {date}, {time}\n')


# now to display the attendance marked we will call this display function in the condition when we press q

def display_attendance():
    # open the csv file and display name, date and time

    try:
        with open(filepath, 'r') as f:
            print("\n -------Attendance Marked Today -------")
            next(f) #to skip the first line, which is header
            
            for line in f:
                name,date,time=line.strip().split(',')

                print(f"Name: {name}, Date: {date}, Time: {time}")

    except FileNotFoundError:
        print("Attendance file not found.")


# 4.start the webcam
cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()

    if not ret:
        break

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # gray is the gray scale image

    # 5. detect faces in the frame
    faces=face_cascade.detectMultiScale(gray,1.1,5)

    for (x,y,w,h) in faces:
        roi_gray=gray[y:y+h,x:x+w]
        # here the cropped image is saved

        id,confidence=recognizer.predict(roi_gray)
        # the model returns the confidence and the id that exists, when an image is identified, 
        # but if it is not sure that the image belongs to paritcular person, then it will return a valid id but with high confidence, 
        # higher the confidence, worst is the match found, and we can predict it as unknown, hence we add a if else 

        if confidence<60:
            name=labels.get(id,"Unknown")
            text=f"{name} ({round(100-confidence,2)}%)"
            # here we r rounding of the accurrary of match % to 2 digits

            # as soon as we have got the name we call the function to mark the attendance
            mark_attendance(name)

        else:
            text="Unknown"

        # 7.Draw rectangle and put text
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

    # 8.Show the frame
    cv2.imshow("Face Recognition",frame)

    if cv2.waitKey(1)==ord('q'):
        display_attendance()
        break

cap.release()
cv2.destroyAllWindows()


