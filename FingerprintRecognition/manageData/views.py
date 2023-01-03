
import os, shutil
import shutil
from time import localtime, strftime
from tkinter import filedialog, messagebox
from django.shortcuts import  render, redirect

from .models import Employee, Image
from .forms import NewEmployeeForm, chooseEmployeetForm, showSampleForm, ImageForm
from django.contrib import messages
from django.contrib.auth.decorators import login_required

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from PIL import Image as ImagePIL
import random
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
import shutil
from PIL import ImageFile

####### Khai bao bien de train data
ImageFile.LOAD_TRUNCATED_IMAGES = True
ImagePIL.MAX_IMAGE_PIXELS = 1000000000
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
root_path = ''
datasetFolderName = root_path+'data'
sourceFiles = []
MODEL_FILENAME=root_path+"model_cv.h5"
classLabels = []
path = 'data/train'
for _, dirnames, filenames in os.walk(path):
    classLabels=dirnames
    break
img_rows, img_cols = 100, 100
train_path=datasetFolderName+'/train/'
test_path=datasetFolderName+'/test/'
validation_path=datasetFolderName+'/validation/'
batch_size = 12
epoch=2
activationFunction='elu'

#--------------------

def homepage(request):
    return render(request, 'home.html')

@login_required
def all_employee(request):
    employee_list = Employee.objects.all()
    return render(request, 'list_employees.html',
                  {'employee_list' : employee_list})

@login_required
def dashboard(request):
    return render(request, 'dashboard.html' )

@login_required
def registryNewEml(request):
	if request.method=='POST':
		form = NewEmployeeForm(request.POST)
		if form.is_valid():
			employee = form.save()
			messages.success(request, f"Registration successful !")
			return redirect('list-employees')
		else :
			return render(request, 'register.html', context={"register_form":form})
	
	form = NewEmployeeForm()
	return render(request, 'register.html', context={"register_form":form})

@login_required
def update_employee(request, employee_id):
    employee = Employee.objects.get(pk=employee_id)
    form = NewEmployeeForm(request.POST or None, instance=employee)
    if form.is_valid():
        employee = form.save()
        messages.success(request, f"Registration successful !")
        return redirect('list-employees')
    else : return render(request, 'update_employee.html', {"employee":employee,"form" : form})
    
    

@login_required
def delete_employee(request, employee_id):
    employee = Employee.objects.get(pk=employee_id)
    employee.delete()
    return redirect('list-employees')
    

@login_required
def choose_employee(request):
    if request.method == 'POST':
        form = chooseEmployeetForm(request.POST)
        data = request.POST.copy()
        id= data.get('employee_id')
        print(type(id))
        if Employee.objects.filter(id=id).exists():
            return redirect('show-sample')
        else :
            messages.warning(request, f'No such employee id found. Please register employee first.')
            return redirect('dashboard')
    else :
        form = chooseEmployeetForm()
        return render(request,'choose_employee.html', {'form': form})
        
@login_required
def show_sample(request, employee_id):
    form= ImageForm(request.POST)
    if Image.objects.filter(fk=employee_id).exists():
        images = Image.objects.all(fk=employee_id)
        return render(request,'show_sample.html', { "form" : form, 'employee_id':employee_id})
    return render(request,'show_sample.html', { 'form':form, 'employee_id':employee_id})
    # images = Image.objects.all()
    # form= showSampleForm(request.POST)
    # return render(request,'show_sample.html', { 'images': images})
      
@login_required
def show_sample_employee(request, employee_id):
    images = Image.objects.get(fk=employee_id)
    return render(request, 'show_sample_employee.html',{'images':images})
      
@login_required      
def add_sample(request, employee_id):
    if request.method == 'POST':
        form = ImageForm(request.POST)
        files = request.FILES.getlist("image")
        if form.is_valid():
            e = form.save(commit=False)
            e.save()
            for i in files:
                Image.objects.create(employee=e)
            return redirect('show-sample')
        else:
            messages.error(request, "Import image file")
            return render(request,'add_sample.html', {employee_id:employee_id, 'form':form})
    else :
        form = ImageForm()
        return render(request,'add_sample.html', {employee_id:employee_id, 'form':form})
        

@login_required
def train(request):
    move_data()
    accuracy, precision, recall, f1Score = train_data()
    accuracy, precision, recall, f1Score = round(accuracy, 2), round(precision, 2), round(recall, 2), round(f1Score, 2)  
    messages.success(request, f"Train data successfull! ")
    messages.success(request,  f"Accuracy = {accuracy}" )
    messages.success(request,  f"Precision = {precision}" )
    messages.success(request,  f"Recall = {recall}" )
    messages.success(request,  f"F1-Score = {f1Score}" )
    return render(request,"train.html")
   
    
def get_last_time_train():
    lastTrained = os.path.getmtime('model_cv.h5') 
    modifiedTime = strftime('%Y-%m-%d %H:%M:%S', localtime(lastTrained))  
    return modifiedTime
  
def get_total_employee():
    path = 'data/test/'
    count=0
    for _, dirnames, filenames in os.walk(path):
        count = len(dirnames)
        break
    return count
  
def get_total_sample():
    train_path = "data/train/"
    test_path = "data/test/"

    num=0
    for _, dirnames, filenames in os.walk(train_path):
        num += len(filenames)
    for _, dirnames, filenames in os.walk(test_path):
        num += len(filenames)
    return num    
  
@login_required
def view_statitics_home(request):
    last_time_train = get_last_time_train()
    total_employee = get_total_employee()
    total_sample = get_total_sample()
    return render(request, 'view_statitics_home.html', {'last_time_train': last_time_train, 
                                                       'total_employee' : total_employee, 
                                                       'total_sample' :total_sample})
  

def create_dataset():
    images = Image.objects.all()
    for image in images:
        id = image.employee.id
        if (os.path.exists('data/train/{}'.format(id))==False):
            os.makedirs('data/train/{}'.format(id))
        if (os.path.exists('data/test/{}'.format(id))==False):
            os.makedirs('data/test/{}'.format(id))
        if (os.path.exists('data/validation/{}'.format(id))==False):
            os.makedirs('data/validation/{}'.format(id))
    
def move_data():
    create_dataset()
    old_dir = "/media/training_data/"
    images = Image.objects.all()
    for image in images:
        image.save()
        shutil.copy(image.image.url[1::], f"data/train/{image.employee.id}")

def transferBetweenFolders(source, dest, splitRate): 
    global sourceFiles
    sourceFiles=os.listdir(source)
    if(len(sourceFiles)!=0):
        transferFileNumbers=int(len(sourceFiles)*splitRate)
        transferIndex=random.sample(range(0, len(sourceFiles)), transferFileNumbers)
        for eachIndex in transferIndex:
            shutil.move(source+str(sourceFiles[eachIndex]), dest+str(sourceFiles[eachIndex]))
    else:
        print("No file moved. Source empty!")
        
def transferAllClassBetweenFolders(source, dest, splitRate):
    for label in classLabels:
        transferBetweenFolders(datasetFolderName+'/'+source+'/'+label+'/', 
                               datasetFolderName+'/'+dest+'/'+label+'/', 
                               splitRate)

def my_metrics(y_true, y_pred):
    accuracy=accuracy_score(y_true, y_pred)
    precision=precision_score(y_true, y_pred,average='weighted')
    recall = recall_score(y_true, y_pred,average='weighted')
    f1Score=f1_score(y_true, y_pred, average='weighted') 
    print("Accuracy  : {}".format(accuracy))
    print("Precision : {}".format(precision))
    print("Recall : {}".format(recall))
    print("f1Score : {}".format(f1Score))
    cm=confusion_matrix(y_true, y_pred)
    print(cm)
    return  accuracy,precision, recall, f1Score


def prepareNameWithLabels(folderName, X, Y):
    sourceFiles=os.listdir(datasetFolderName+'/train/'+folderName)
    for val in sourceFiles:
        X.append(val)
        for i in range(len(classLabels)):
          if(folderName==classLabels[i]):
              Y.append(i)
              
#chuan bi data
def preprocess(X, Y):
    transferAllClassBetweenFolders('test', 'train', 1.0)
    transferAllClassBetweenFolders('train', 'test', 0.3)
    for i in range(len(classLabels)):
        prepareNameWithLabels(classLabels[i], X, Y)
  

#Model
# Note that, this model structure is a very basic one. To achieve better performance, you should change the model structure and hyperparameters according to your needs and data.
def getModel():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', activation=activationFunction, input_shape=(img_rows, img_cols, 3)))
    model.add(Conv2D(64, (3, 3), activation=activationFunction))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(32, (3, 3), padding='same', activation=activationFunction))
    model.add(Conv2D(32, (3, 3), activation=activationFunction))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(16, (3, 3), padding='same', activation=activationFunction))
    model.add(Conv2D(16, (3, 3), activation=activationFunction))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(64, activation=activationFunction))
    model.add(Dropout(0.1))                  
    model.add(Dense(32, activation=activationFunction))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation=activationFunction))
    model.add(Dropout(0.1))
    model.add(Dense(len(classLabels), activation='softmax')) 
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model




#======================Train
def train_data():
    model=getModel()
    X=[]
    Y=[]
    transferAllClassBetweenFolders('validation', 'train', 1.0)
    preprocess(X, Y)
    X=np.asarray(X)
    Y=np.asarray(Y)
    skf = StratifiedKFold(n_splits=2, shuffle=True)
    skf.get_n_splits(X, Y)
    foldNum=0
    for train_index, val_index in skf.split(X, Y):
    #First cut all images from validation to train (if any exists)
        
        foldNum+=1
        print("Results for fold",foldNum)
        
        train_datagen = ImageDataGenerator(
                      rescale=1./255,
                      zoom_range=0.20,
                      fill_mode="nearest")
        validation_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
    #Start ImageClassification Model
        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training')

        validation_generator = validation_datagen.flow_from_directory(
            validation_path,
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode=None,  # only data, no labels
            shuffle=False)   
   
    # fit model
        history=model.fit(train_generator, 
                        epochs=2)
    
    print("==============TEST RESULTS============")
    test_generator = test_datagen.flow_from_directory(
            test_path,
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False) 
    predictions = model.predict(test_generator, verbose=1)
    yPredictions = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes

    testAcc, testPrecision, testRecall, testF1Score= my_metrics(true_classes, yPredictions)
    # model.save(MODEL_FILENAME)
    
    
    return testAcc, testPrecision, testRecall, testF1Score