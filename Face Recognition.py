#Face Recognition
import numpy as np
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier

# Recognise Faces using some classification algorithm - like Logistic, KNN, SVM etc.
# 1. load the training data (numpy arrays of all the persons)
		# x- values are stored in the numpy arrays
		# y-values we need to assign for each person
# 2. Read a video stream using opencv
# 3. extract faces out of it
# 4. use knn to find the prediction of face (int)
# 5. map the predicted id to name of the user 
# 6. Display the predictions on the screen - bounding box and name

#knn algorithm
def knn(train, test, k=5):
    # Split the training data into features and labels
    X_train = train[:, :-1]
    y_train = train[:, -1]
    
    # Create the KNN model
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Fit the model with the training data
    knn.fit(X_train, y_train)
    
    # Predict the label of the test data
    prediction = knn.predict([test])
    
    return prediction[0]


cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('C:/Users/Satanya/Desktop/Image Processing/Face Detection Project/haarcascade_frontalface_alt.xml')

dataset_path = 'C:/Users/Satanya/Desktop/Image Processing/Face Detection Project/'

face_data = []
labels = []
class_id = 0
names = {}

#Dataset Preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-3]
        data_item = np.load(os.path.join(dataset_path , fx))
        face_data.append(data_item)

        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
print(face_labels.shape)
print(face_dataset.shape)

trainset = np.concatenate((face_dataset, face_labels), axis=1)

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if ret == False:
        continue
	# Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect multi faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for face in faces:
        x, y, w, h = face

		# Get the face ROI
        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

        out = knn(trainset, face_section.flatten())

		# Draw rectangle in the original image
        cv2.putText(frame, names[int(out)],(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255), 2)

    cv2.imshow("Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


