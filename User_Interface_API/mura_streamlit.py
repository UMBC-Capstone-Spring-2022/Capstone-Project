# -*- coding: utf-8 -*-


#pip install streamlit

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import base64
import datetime
import os
from PIL import Image
from tqdm import tqdm
import pickle
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
if torch.cuda.is_available():
    print('CUDA is available. Working on GPU')
    DEVICE = torch.device('cuda')
else:
    print('CUDA is not available. Working on CPU')
    DEVICE = torch.device('cpu')

app_mode = st.sidebar.selectbox('Select Page',['Home','Prediction'])

data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

#Frontend

if app_mode=='Home':
    st.header('MURA Predictions')
    st.image("mura.png")
    

else:
  category=st.selectbox('Image category',['Forearm', 'Wrist', 'Shoulder', 'Finger','Hand', 'Humerus','Elbow'])  
  image= st.file_uploader('upload the image', type=['png'])
  
  if image is not None:
      st.image(image, caption='Uploaded Image.', use_column_width=False)
      if st.button('Predict'):
          if category=='Forearm':
              loaded_model_forearm = models.densenet161(pretrained=True)

              for param in loaded_model_forearm.parameters():
                  param.requires_grad = False
              loaded_model_forearm.classifier = torch.nn.Linear(loaded_model_forearm.classifier.in_features, out_features=200)
              loaded_model_forearm.load_state_dict(torch.load("Forearm_re_DenseNet161_weights.pth",map_location=torch.device('cpu'))) 
              loaded_model_forearm.to('cpu')

              def forearm_predict(img):
                  img_pil = Image.open(img)
                  img_trans = data_transforms(img_pil)
                  img_trans.to('cpu')
                  img_trans = img_trans.unsqueeze(0)
                  y_pred = loaded_model_forearm(img_trans)
                  f_prediction = torch.argmax(y_pred, 1)
                  f_output= f_prediction.item()
                  return f_output
              
              prediction = str(forearm_predict(image))
              
          elif category=='Hand':

              loaded_model_hand = models.densenet161(pretrained=True)

              for param in loaded_model_hand.parameters():
                  param.requires_grad = False
              loaded_model_hand.classifier = torch.nn.Linear(loaded_model_hand.classifier.in_features, out_features=200)
              loaded_model_hand.load_state_dict(torch.load("Hand_re_DenseNet161_weights.pth",map_location=torch.device('cpu'))) 
              loaded_model_hand.to('cpu')

              def hand_predict(img):
                  img_pil = Image.open(img)
                  img_trans = data_transforms(img_pil)
                  img_trans.to('cpu')
                  img_trans = img_trans.unsqueeze(0)
                  y_pred = loaded_model_hand(img_trans)
                  h_prediction = torch.argmax(y_pred, 1)
                  h_output= h_prediction.item()
                  return h_output

              prediction = str(hand_predict(image))
              
          elif category=='Humerus':

              loaded_model_humerus = models.densenet161(pretrained=True)

              for param in loaded_model_humerus.parameters():
                  param.requires_grad = False
              loaded_model_humerus.classifier = torch.nn.Linear(loaded_model_humerus.classifier.in_features, out_features=200)
              loaded_model_humerus.load_state_dict(torch.load("Humerus.pth",map_location=torch.device('cpu'))) 
              loaded_model_humerus.to('cpu')

              def humerus_predict(img):
                  img_pil = Image.open(img)
                  img_trans = data_transforms(img_pil)
                  img_trans.to('cpu')
                  img_trans = img_trans.unsqueeze(0)
                  y_pred = loaded_model_humerus(img_trans)
                  hu_prediction = torch.argmax(y_pred, 1)
                  hu_output= hu_prediction.item()
                  return hu_output

              prediction = str(humerus_predict(image))
              
          elif category=='Finger':

              loaded_model_finger = models.densenet161(pretrained=True)

              for param in loaded_model_finger.parameters():
                  param.requires_grad = False
              loaded_model_finger.classifier = torch.nn.Linear(loaded_model_finger.classifier.in_features, out_features=200)
              loaded_model_finger.load_state_dict(torch.load("Finger_re_DenseNet161_weights.pth",map_location=torch.device('cpu'))) 
              loaded_model_finger.to('cpu')

              def finger_predict(img):
                  img_pil = Image.open(img)
                  img_trans = data_transforms(img_pil)
                  img_trans.to('cpu')
                  img_trans = img_trans.unsqueeze(0)
                  y_pred = loaded_model_finger(img_trans)
                  Fi_prediction = torch.argmax(y_pred, 1)
                  Fi_output= Fi_prediction.item()
                  return Fi_output

              prediction = str(finger_predict(image))
              
          elif category=='Wrist':

              loaded_model_wrist = models.densenet161(pretrained=True)

              for param in loaded_model_wrist.parameters():
                  param.requires_grad = False
              loaded_model_wrist.classifier = torch.nn.Linear(loaded_model_wrist.classifier.in_features, out_features=200)
              loaded_model_wrist.load_state_dict(torch.load("Wrist.pth",map_location=torch.device('cpu'))) 
              loaded_model_wrist.to('cpu')

              def wrist_predict(img):
                  img_pil = Image.open(img)
                  img_trans = data_transforms(img_pil)
                  img_trans.to('cpu')
                  img_trans = img_trans.unsqueeze(0)
                  y_pred = loaded_model_wrist(img_trans)
                  W_prediction = torch.argmax(y_pred, 1)
                  W_output= W_prediction.item()
                  return W_output

              prediction = str(wrist_predict(image))
              
          elif category=='Elbow':

              loaded_model_elbow = models.densenet161(pretrained=True)

              for param in loaded_model_elbow.parameters():
                  param.requires_grad = False
              loaded_model_elbow.classifier = torch.nn.Linear(loaded_model_elbow.classifier.in_features, out_features=200)
              loaded_model_elbow.load_state_dict(torch.load("Elbow.pth",map_location=torch.device('cpu'))) 
              loaded_model_elbow.to('cpu')

              def elbow_predict(img):
                  img_pil = Image.open(img)
                  img_trans = data_transforms(img_pil)
                  img_trans.to('cpu')
                  img_trans = img_trans.unsqueeze(0)
                  y_pred = loaded_model_elbow(img_trans)
                  E_prediction = torch.argmax(y_pred, 1)
                  E_output= E_prediction.item()
                  return E_output

              prediction = str(elbow_predict(image))
              
          else:


              loaded_model_shoulder = models.densenet161(pretrained=True)

              for param in loaded_model_shoulder.parameters():
                  param.requires_grad = False
              loaded_model_shoulder.classifier = torch.nn.Linear(loaded_model_shoulder.classifier.in_features, out_features=200)
              loaded_model_shoulder.load_state_dict(torch.load("Shoulder_re_DenseNet161_weights.pth",map_location=torch.device('cpu'))) 
              loaded_model_shoulder.to('cpu')

              def shoulder_predict(img):
                  img_pil = Image.open(img)
                  img_trans = data_transforms(img_pil)
                  img_trans.to('cpu')
                  img_trans = img_trans.unsqueeze(0)
                  y_pred = loaded_model_shoulder(img_trans)
                  S_prediction = torch.argmax(y_pred, 1)
                  S_output= S_prediction.item()
                  return S_output
              prediction = str(shoulder_predict(image))


            
        
          if prediction=='0':
                output= st.write('normal')
          else:
                output= st.write('abnormal')
