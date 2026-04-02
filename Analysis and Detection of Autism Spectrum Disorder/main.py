
from tkinter import *
from tkinter import filedialog
import tkinter as tk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical
from keras.layers import MaxPooling2D, Dense, Flatten, Convolution2D
from keras.models import Sequential, Model
from keras.layers import Input

# ---------------- MAIN WINDOW ----------------

main = Tk()
main.title("Autism Spectrum Disorder Detection System")
main.geometry("1300x900")
main.configure(bg="#0f172a")

# ---------------- GLOBAL VARIABLES ----------------

accuracy=[]
precision=[]
recall=[]
fscore=[]
specificity=[]
sensitivity=[]

global filename,dataset,X,Y
global X_train,X_test,y_train,y_test
global label_encoder,classifier,hist,columns

# ---------------- TITLE ----------------

title = Label(main,
text="Analysis and Detection of Autism Spectrum Disorder Using Machine Learning",
font=("Segoe UI",22,"bold"),
bg="#020617",
fg="white",
pady=15)

title.pack(fill=X)

# ---------------- FRAME STRUCTURE ----------------

topFrame = Frame(main,bg="#0f172a")
topFrame.pack(pady=10)

middleFrame = Frame(main,bg="#0f172a")
middleFrame.pack()

bottomFrame = Frame(main,bg="#0f172a")
bottomFrame.pack(pady=10)

# ---------------- TEXT OUTPUT ----------------

scrollbar = Scrollbar(bottomFrame)
scrollbar.pack(side=RIGHT,fill=Y)

text = Text(bottomFrame,
height=25,
width=150,
bg="#020617",
fg="#22c55e",
font=("Consolas",11),
yscrollcommand=scrollbar.set)

text.pack()

scrollbar.config(command=text.yview)

# ---------------- BUTTON STYLE ----------------

def buttonStyle(btn):

    btn.configure(
    font=("Segoe UI",11,"bold"),
    bg="#14b8a6",
    fg="white",
    width=30,
    height=2,
    bd=0,
    cursor="hand2")

    btn.bind("<Enter>",lambda e:btn.config(bg="#0d9488"))
    btn.bind("<Leave>",lambda e:btn.config(bg="#14b8a6"))

# ---------------- FUNCTIONS ----------------

def upload():

    global filename,dataset

    filename = filedialog.askopenfilename()

    text.delete('1.0',END)
    text.insert(END,"Dataset Loaded\n\n")

    dataset = pd.read_csv(filename)

    text.insert(END,str(dataset.head())+"\n")

    label = dataset.groupby('Class/ASD').size()

    label.plot(kind="bar")
    plt.title("Autism Distribution")
    plt.show()

# ---------------- PREPROCESS ----------------

def processDataset():

    global X,Y,label_encoder,X_train,X_test,y_train,y_test,columns,dataset

    label_encoder=[]
    dataset.fillna(0,inplace=True)

    columns = dataset.columns

    for i in range(11,len(columns)):
        if i!=17:
            le=LabelEncoder()
            dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))
            label_encoder.append(le)

    dataset = dataset.values

    X = dataset[:,0:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]

    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)

    text.insert(END,"\nDataset Preprocessed Successfully\n")
    text.insert(END,"\nTotal Records : "+str(X.shape[0]))
    text.insert(END,"\nTraining Records : "+str(X_train.shape[0]))
    text.insert(END,"\nTesting Records : "+str(X_test.shape[0]))

# ---------------- METRICS ----------------

def calculateMetrics(name,predict,testY):

    p = precision_score(testY,predict,average='macro')*100
    r = recall_score(testY,predict,average='macro')*100
    f = f1_score(testY,predict,average='macro')*100
    a = accuracy_score(testY,predict)*100

    cm = confusion_matrix(testY,predict)

    text.insert(END,"\n"+name+" Results\n")
    text.insert(END,"\nAccuracy : "+str(a))
    text.insert(END,"\nPrecision : "+str(p))
    text.insert(END,"\nRecall : "+str(r))
    text.insert(END,"\nFScore : "+str(f)+"\n")

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

    sns.heatmap(cm,annot=True,cmap="viridis")
    plt.title(name+" Confusion Matrix")
    plt.show()

# ---------------- KNN ----------------

def runKNN():

    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train,y_train)

    predict = knn.predict(X_test)

    calculateMetrics("KNN",predict,y_test)

# ---------------- ANN ----------------

def runANN():

    ann = MLPClassifier()
    ann.fit(X_train,y_train)

    predict = ann.predict(X_test)

    calculateMetrics("ANN",predict,y_test)

# ---------------- PROPOSED MODEL ----------------

def runProposed():

    km = KMeans(n_clusters=2)
    km.fit(X_train)

    cluster_labels = km.labels_

    X_train_mdc = np.hstack((X_train,cluster_labels.reshape(-1,1)))

    input_dim = X_train_mdc.shape[1]

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(input_dim//2,activation='relu')(input_layer)
    decoded = Dense(input_dim,activation='relu')(encoded)

    autoencoder = Model(input_layer,decoded)
    encoder = Model(input_layer,encoded)

    autoencoder.compile(optimizer='adam',loss='mse')
    autoencoder.fit(X_train_mdc,X_train_mdc,epochs=20,batch_size=8,verbose=0)

    X_train_encoded = encoder.predict(X_train_mdc)

    svm_cls = svm.SVC()
    svm_cls.fit(X_train_encoded,y_train)

    cluster_test = km.predict(X_test)
    X_test_mdc = np.hstack((X_test,cluster_test.reshape(-1,1)))

    X_test_encoded = encoder.predict(X_test_mdc)

    predict = svm_cls.predict(X_test_encoded)

    calculateMetrics("Proposed BDML-MDCASD",predict,y_test)

# ---------------- CONVTRANSNET ----------------

def runConvTransNet():

    global classifier,hist

    X1 = np.reshape(X,(X.shape[0],X.shape[1],1,1))
    Y1 = to_categorical(Y)

    X_train1,X_test1,y_train1,y_test1 = train_test_split(X1,Y1,test_size=0.2)

    classifier = Sequential()

    classifier.add(Convolution2D(32,1,1,input_shape=(X_train1.shape[1],X_train1.shape[2],X_train1.shape[3]),activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(1,1)))

    classifier.add(Convolution2D(32,1,1,activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(1,1)))

    classifier.add(Flatten())
    classifier.add(Dense(256,activation='relu'))
    classifier.add(Dense(y_train1.shape[1],activation='softmax'))

    classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    hist = classifier.fit(X_train1,y_train1,epochs=30,batch_size=16,validation_data=(X_test1,y_test1))

    predict = classifier.predict(X_test1)
    predict = np.argmax(predict,axis=1)
    y_test1 = np.argmax(y_test1,axis=1)

    calculateMetrics("ConvTransNet",predict,y_test1)

# ---------------- PERFORMANCE GRAPH ----------------

def graph():

    plt.bar(["KNN","ANN","Proposed","ConvTransNet"],accuracy)
    plt.title("Algorithm Accuracy Comparison")
    plt.show()

# ---------------- TRAINING GRAPH ----------------

def ConvTransNetgraph():

    history = hist.history

    plt.plot(history['accuracy'])
    plt.plot(history['loss'])

    plt.title("ConvTransNet Training Graph")
    plt.legend(["Accuracy","Loss"])
    plt.show()

# ---------------- DETECT AUTISM ----------------

def detectAutism():
    global classifier, label_encoder, columns
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    testData = pd.read_csv(filename)
    testData.fillna(0, inplace=True)
    testData = testData.replace(np.nan, 0)
    columns = testData.columns
    j = 0

    for i in range(11, len(columns)):
        if i != 17:
            testData[columns[i]] = pd.Series(label_encoder[j].transform(testData[columns[i]].astype(str)))
            j += 1

    testData = testData.values
    X1 = np.reshape(testData, (testData.shape[0], testData.shape[1], 1, 1))
    predict = classifier.predict(X1)
    predict = np.argmax(predict, axis=1)
    label = ["No Autism Disorder Detected", "Autism Disorder Detected"]

    for i in range(len(predict)):
        text.insert(END, "Test Data = " + str(testData[i]) + " => " + label[predict[i]] + "\n\n")


# ---------------- BUTTONS ----------------

btn1=Button(topFrame,text="Upload ASD Dataset",command=upload)
btn1.grid(row=0,column=0,padx=10,pady=10)

btn2=Button(topFrame,text="Preprocess Dataset",command=processDataset)
btn2.grid(row=0,column=1,padx=10,pady=10)

btn3=Button(topFrame,text="Run KNN Algorithm",command=runKNN)
btn3.grid(row=0,column=2,padx=10,pady=10)

btn4=Button(topFrame,text="Run ANN Algorithm",command=runANN)
btn4.grid(row=1,column=0,padx=10,pady=10)

btn5=Button(topFrame,text="Run Proposed BDML-MDCASD",command=runProposed)
btn5.grid(row=1,column=1,padx=10,pady=10)

btn6=Button(topFrame,text="Run ConvTransNet",command=runConvTransNet)
btn6.grid(row=1,column=2,padx=10,pady=10)

btn7=Button(middleFrame,text="Detect Autism from Test Data",command=detectAutism)
btn7.grid(row=0,column=0,padx=10,pady=10)

btn8=Button(middleFrame,text="All Algorithms Performance Graph",command=graph)
btn8.grid(row=0,column=1,padx=10,pady=10)

btn9=Button(middleFrame,text="ConvTransNet Training Graph",command=ConvTransNetgraph)
btn9.grid(row=0,column=2,padx=10,pady=10)

# Apply style

for widget in topFrame.winfo_children():
    buttonStyle(widget)

for widget in middleFrame.winfo_children():
    buttonStyle(widget)

main.mainloop()
