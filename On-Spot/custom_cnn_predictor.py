import cv2
import numpy as np
import pickle
with open('/kaggle/input/mmodelpkl/mmodel.pkl', 'rb') as file:
    model = pickle.load(file)
target_size = (150, 150)
def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    return image
image_path = '/kaggle/input/images0/images.jpeg'
preprocessed_image = preprocess_image(image_path, target_size)
def final()
input_image = np.expand_dims(preprocessed_image, axis=0)
predictions = model.predict(input_image)
predicted_class = np.argmax(predictions)
tumor_type = ["glioma_tumor", 'meningioma_tumor',"no_tumor",'pituitary_tumor']
print(tumor_type[predicted_class])
