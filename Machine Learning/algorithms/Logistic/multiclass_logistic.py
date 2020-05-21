import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
import seaborn as sb
digits=load_digits()

#check the data
#print(dir(digits))

#check the image
#plt.gray()
#plt.show(plt.matshow(digits.images[5]))
X_train,X_test,y_train,y_test = train_test_split(digits.data,digits.target , test_size=0.2)
#print(len(X_train))

model=LogisticRegression()
model.fit(X_train,y_train)
dat1=model.predict(X_test)


#for confusion matrix
cm=confusion_matrix(y_test,dat1)
plt.figure(figsize=(10,5))
plt.show(sb.heatmap(cm,annot=True))
plt.xlabel('predicted')
plt.ylabel('Truth')
