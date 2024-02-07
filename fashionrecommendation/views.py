from django.shortcuts import render
import os
from django.core.files.storage import FileSystemStorage

from .models import UploadImage
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf 
import h5py
import keras
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.applications.resnet50  import preprocess_input
from tensorflow.keras.models import load_model

# Create your views here.

def imageinput(request):
    images=[]
    
    if request.method=="POST":
        image=request.FILES["image"]
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        image_path = fs.path(filename)
        image=load_img(image_path,target_size=(224,224,3))
        image=img_to_array(image)
        image=np.expand_dims(image,axis=0)
        preprocessedimage=preprocess_input(image)
        
        model=load_model("/Users/khumapokharel/Desktop/geekyshowsagiserver/FashionRecommendationSystem/recommandor/models/model.h5")
        
        # model = keras.models.load_model("/Users/khumapokharel/Desktop/geekyshowsagiserver/FashionRecommendationSystem/recommandor/models/model.h5")
        # model=load_model("/Users/khumapokharel/Desktop/geekyshowsagiserver/FashionRecommendationSystem/recommandor/models/model.h5")
        
        result=model.predict(preprocessedimage).flatten()
        
        normalizedresult=result/norm(result)
        
        extraced_features=np.array(pickle.load(open("/Users/khumapokharel/Desktop/geekyshowsagiserver/FashionRecommendationSystem/recommandor/models/extracted_feature.pkl","rb")))
        
        filenames=np.array(pickle.load(open("/Users/khumapokharel/Desktop/geekyshowsagiserver/FashionRecommendationSystem/recommandor/models/images.pkl","rb")))
        neighbours=NearestNeighbors(n_neighbors=6,algorithm='brute',metric="euclidean")
        neighbours.fit(extraced_features)
        distances,indeces=neighbours.kneighbors([normalizedresult])
        for i in indeces:
            for k in i:
                images.append(filenames[k])
       
        
        
        
        
    
    return render(request,"fashion.html",{"imagespath":images})


    





