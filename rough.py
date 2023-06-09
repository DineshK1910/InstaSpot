import cv2
import numpy as np
import pickle
import tensorflow as tf
#with open('Model.pkl', 'rb') as file:
#    model = pickle.load(file)
new_model = tf.keras.models.load_model('FinalModel.h5')
target_size = (150, 150)


def preprocess_image(image_path, target_size):
    
    image=cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    input_image = np.expand_dims(image, axis=0)
    predictions = new_model.predict(input_image)
    predicted_class = np.argmax(predictions)
    tumor_type = ["glioma_tumor", 'meningioma_tumor',
                  "no_tumor", 'pituitary_tumor']
    return (tumor_type[predicted_class])


# path = 'bt1.jpg'
# print(preprocess_image(path, target_size))
