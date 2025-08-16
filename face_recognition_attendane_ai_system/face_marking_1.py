# for computer vision tasks we import OpenCV library  ---->>computer vision works- face detection, recognition, image resizing croping etc
import cv2

#  loading the haar cascade face detector from opencv
# this xml file contains the trained face detection model
face_cascade = cv2.CascadeClassifier(
    'c:/Users/gawan/BTECH 2 Year/Python Projects 2025/face_recognition_attendane_ai_system/haarcascade_frontalface_default.xml'
)


# intialize the webcam, the default camera is 0
cap=cv2.VideoCapture(0)


# now, we loop to read frames from the webcamera continuously
while True:
    
    # capture a single frame from web cam
    ret,frame=cap.read()

    # ret is a boolean variable which returns True when the frame is read properly
    #frame is the actual image captured, frame is a numpy array (multidimensional) (representation of a single image as numpy array)

    if not ret:
        break  #if the frame is not read properly, break the loop


    # now, we have to convert the colorful image to grayscale
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect the faces in the grayscale frame
    # detectMultiscale returns a list of coordinates and sizes of the rectangles(numpy array)
    faces=face_cascade.detectMultiScale(
        gray, #input image variable
        scaleFactor=1.1, #how much the image size is reduced at each scale
        minNeighbors=5 # how many neighbors each rectangle should have to be considered a face
    )

    # the faces contains the array of coordinates (x,y,w,h)

    # loop through each detected face and draw a rectangle around it
    for(x,y,w,h) in faces:
        
        # draw rectangle on the orginal colored frame
        # (x,y) is the top-left point, (x+w,y+h) is the bottom right
        #we color the rectangle with blue its rgb equivalent is : (255,0,0), 
        #2 is the thickness of the rectangle
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    # show the frame with rectangles in a window
    cv2.imshow("Face Detection AI", frame)


        # wait for 1ms between frames and check if 'q' is pressed
    if cv2.waitKey(1)==ord('q'):
        # if q is pressed break the loop
        break

# release the webcam 
cap.release()

# close the opencv windows opened during the program
cv2.destroyAllWindows()

    

