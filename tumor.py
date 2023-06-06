import os 
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
tumor_type = ["glioma_tumor", 'meningioma_tumor',"no_tumor",'pituitary_tumor']
data_path='/home/dinesh/Downloads/kaggle/brain_tumor/Training'
xTrain=[]
yTrain=[]
for i in tumor_type:
    path=os.path.join('/home/dinesh/Downloads/kaggle/brain_tumor/Training',i)
    for j in os.listdir(path):
        image=cv2.imread(os.path.join(path,j))
        image=cv2.resize(image,(124,124))
        xTrain.append(image)
        yTrain.append(tumor_type.index(i))
xTrain=np.array(xTrain)
yTrain=np.array(yTrain)
print(xTrain.shape)
print(yTrain.shape)
xtr,xts,ytr,yts=train_test_split(xTrain,yTrain,test_size=0.2,random_state=21)
print("Splitting Done")
svm=SVC(kernel='rbf')
svm.fit(xtr.reshape(xtr.shape[0],-1),ytr.reshape(-1))
y_pred=svm.predict(xts.reshape(xts.shape[0],-1))
acc=accuracy_score(yts,y_pred)
print("Accuracy:",acc)
