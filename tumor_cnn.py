import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout
from tensorflow.keras.utils import to_categorical


target_size = (150, 150)

trdataset_path = "Training"
tsdataset_path = "Testing"

classes = ["glioma_tumor",'meningioma_tumor', "no_tumor",'pituitary_tumor']


def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    return image

X = []
y = []

for class_idx, class_name in enumerate(classes):
    class_folder = os.path.join(trdataset_path, class_name)
    for filename in os.listdir(class_folder):
        image_path = os.path.join(class_folder, filename)
        preprocessed_image = preprocess_image(image_path, target_size)
        X.append(preprocessed_image)
        y.append(class_idx)
for class_idx, class_name in enumerate(classes):
    class_folder = os.path.join(tsdataset_path, class_name)
    for filename in os.listdir(class_folder):
        image_path = os.path.join(class_folder, filename)
        preprocessed_image = preprocess_image(image_path, target_size)
        X.append(preprocessed_image)
        y.append(class_idx)

X = np.array(X)
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


num_classes = len(classes)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(target_size[0], target_size[1], 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


model.fit(X_train, y_train, batch_size=32, epochs=15, validation_data=(X_test, y_test))

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

print(classification_report(y_test_classes, y_pred_classes, target_names=classes))
pickle.dump(model,open('mmodel.pkl','wb'))
model.save('FinalModel.h5')