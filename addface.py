import cv2
import pickle
import os
import numpy as np

#create an empty list to store the face_data
face_data = []
#counter to keep track of the frames captured
i = 0
#start the videocapture
cam = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
#enter the name you want to save for the face
name = input("Enter the name--->")
# Start capturing frames from the webcam
ret = True
while(ret):
    # Read a frame from the webcam
    ret, frame = cam.read()
    if ret == True:
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the grayscale frame
        face_coordinates = cascade.detectMultiScale(gray, 1.3, 4)
        for (x,y,w,h) in face_coordinates:
            # Extract the region of interest (face) from the frame
            faces = frame[y:y+h, x:x+w, :]
            # Resize the face to a standard size (50x50)
            resized_faces = cv2.resize(faces, (50, 50))
            #Append our frame into face_data list for every 10 frames
            if i % 10 == 0 and len(face_data) < 100: #we wanted to capture 100frames
                face_data.append(resized_faces)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Increment the frame counter
        i += 1
        # Display the frame with rectangles around detected faces
        cv2.imshow('frames', frame)
        # Break the loop if the 'Esc' key is pressed or if 100 faces are captured
        if cv2.waitKey(1) == 27 or len(face_data) >= 100:
            break
    else:
        print('error')
        break
cv2.destroyAllWindows()
cam.release() #release the camera capture

#convert the face_data into array
face_data = np.asarray(face_data)
face_data = face_data.reshape(100, -1)

#create faces.pkl and names.pkl files
if 'names.pkl' not in os.listdir('data/'):
    # If not, create a list of names for the current user (repeated 100 times)
    names = [name]*100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    # Append the current user's name to the names list (repeated 100 times)
    names = names + [name]*100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

if 'faces.pkl' not in os.listdir('data/'):
    with open('data/faces.pkl', 'wb') as w:
        pickle.dump(face_data, w)
else:
    with open('data/faces.pkl', 'rb') as w:
        faces = pickle.load(w)
    # Append the current user's face data to the faces array
    faces = np.append(faces, face_data, axis=0)
    # Save the updated faces array to 'data/faces.pkl'
    with open('data/faces.pkl', 'wb') as w:
        pickle.dump(faces, w)




            
            
            



        
        
        





