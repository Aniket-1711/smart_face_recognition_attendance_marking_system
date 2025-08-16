# Importing necessary Libraries

import cv2 #pip install oopencv-python
import os
import numpy as np
from PIL import Image  #pip install pillow

# step 1: setting the path where all the images are stored

dataset_path="C:/Users/gawan/BTECH 2 Year/Python Projects 2025/smart_face_recognition_attendance_marking_system/face_recognition_attendane_ai_system/dataset"

# step 2: initialize the LBPH face recognizer
# LBPH: local binary patterns histograms
recognizer=cv2.face.LBPHFaceRecognizer_create()

# step 3: Load the Haar cascade for face detection
face_cascade=cv2.CascadeClassifier("C:/Users/gawan/BTECH 2 Year/Python Projects 2025/smart_face_recognition_attendance_marking_system/face_recognition_attendane_ai_system/haarcascade_frontalface_default.xml")

# step 4: create a function to get image and their labels
def get_images_and_labels(path):
    faces_samples=[]  #to store cropped face images
    ids=[]  # to store labels (ids)
    names={} #dictionary to map names to numeric labels
    current_id=0    # to assign unique ids


    # step 5: loop through all files in the dataset folder
    for image_name in os.listdir(path):
        image_path=os.path.join(path,image_name)
        

        # step 6: open the image and convert to grayscale
        pil_image=Image.open(image_path).convert('L')
        # we r converting the image again to grayscale despite saving it in the grayscale format because when we open the the image or using cv2.read() also the grayscale image is converted to color image, which the opencv isn't capable to detect faces, so we need to convert it back to grayscale

        image_np=np.array(pil_image,dtype='uint8')
        # we need to convert it to array because the opencv needs images are array of numbers to process mathematically

        # step 7: collect the first part of the name of the image
        name=image_name.split('_')[0]

        # step 8: check if the name already exists in the dictionary, if not then add and assign a unique id

        if name not in names:
            names[name]=current_id
            current_id+=1

        # now collect the id, if the first part of the name is same, the same id will be mapped to all the images of the same person (50 photos, differnt names but same id)
        # this helps the model to train 

        id_=names[name]

        # step 9: detect the faces (again, to check maintain the consistency and accuracy that it is a face)
        faces=face_cascade.detectMultiScale(image_np)

        for(x,y,w,h) in faces:
            face=image_np[y:y+h,x:x+w]
            faces_samples.append(face)
            ids.append(id_)

    return faces_samples,ids,names


# step 10: call the function to get training data
print("[INFO] Training faces.Please wait...")
faces,ids,name_map=get_images_and_labels(dataset_path)

# step 11: train the recognizer on the faces and labels
recognizer.train(faces,np.array(ids))
# train method needs data in numpy array format
# faces are list of grayscale images
# labels are list of corresponding labels
#  converting to arrays = required by OpenCV + faster processing


# step 12: save the trained model
recognizer.save("C:/Users/gawan/BTECH 2 Year/Python Projects 2025/smart_face_recognition_attendance_marking_system/face_recognition_attendane_ai_system/trainer.yml")
# saving the trained model in yml format so need not train it again and again, and we can use by read() function

# step 13: save the name-id mapping as a refernce 
with open("C:/Users/gawan/BTECH 2 Year/Python Projects 2025/smart_face_recognition_attendance_marking_system/face_recognition_attendane_ai_system/name_labels.txt", 'w') as f:
    for name,id_ in name_map.items():
        f.write(f'{id_}:{name}\n')

print(f"[INFO] Training complete. Model saved as 'trainer.yml'")
print(f"[INFO] Name-ID mapping saved as 'name_labels.txt'")
