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
from transformers import ViTForImageClassification, ViTFeatureExtractor
from datasets import load_metric, Dataset
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

              device = "cuda" if torch.cuda.is_available() else "cpu"
              
              def predict_class(img, model_path, device):
    
                    # feature extractor to make image transforms same as pre-trained model
                    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
                    
                    # process the image according to vit and return tensor
                    def process_image(img):
                        img = Image.open(img).convert('RGB')
                        pixels = feature_extractor(img, return_tensors='pt')['pixel_values']
                        return pixels

                    # process image and send to device
                    img = process_image(img)
                    img = img.to(device)

                    # label dicts
                    target_to_label_2 = {'normal' : 0, 'abnormal': 1}
                    label_to_target_2 = {0 : 'normal', 1 : 'abnormal'}

                    # create ViT model for predictions
                    ViT_test_model = ViTForImageClassification.from_pretrained(
                    model_path,
                    num_labels=2,
                    id2label=label_to_target_2,
                    label2id=target_to_label_2
                    )

                    # move model to device
                    ViT_test_model.to(device)

                    # do a forward pass and get logits
                    with torch.no_grad():
                        outputs = ViT_test_model(img)
                        logits = outputs.logits

                    # get prediction and translate id2label
                    prediction = logits.argmax(-1)
                    label = ViT_test_model.config.id2label[prediction.item()]

                    return label

              
              humerus_pred_abnorm = predict_class(image, 'Humerus', device)
              
              prediction = str(humerus_pred_abnorm)

              
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
              device = "cuda" if torch.cuda.is_available() else "cpu"
              
              def predict_class(img, model_path, device):
    
                    # feature extractor to make image transforms same as pre-trained model
                    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
                    
                    # process the image according to vit and return tensor
                    def process_image(img):
                        img = Image.open(img).convert('RGB')
                        pixels = feature_extractor(img, return_tensors='pt')['pixel_values']
                        return pixels

                    # process image and send to device
                    img = process_image(img)
                    img = img.to(device)

                    # label dicts
                    target_to_label_2 = {'normal' : 0, 'abnormal': 1}
                    label_to_target_2 = {0 : 'normal', 1 : 'abnormal'}

                    # create ViT model for predictions
                    ViT_test_model = ViTForImageClassification.from_pretrained(
                    model_path,
                    num_labels=2,
                    id2label=label_to_target_2,
                    label2id=target_to_label_2
                    )

                    # move model to device
                    ViT_test_model.to(device)

                    # do a forward pass and get logits
                    with torch.no_grad():
                        outputs = ViT_test_model(img)
                        logits = outputs.logits

                    # get prediction and translate id2label
                    prediction = logits.argmax(-1)
                    label = ViT_test_model.config.id2label[prediction.item()]

                    return label

              
              wrist_pred_abnorm = predict_class(image, 'Wrist', device)
              
              prediction = str(wrist_pred_abnorm)
              
          elif category=='Elbow':

              device = "cuda" if torch.cuda.is_available() else "cpu"
              
              def predict_class(img, model_path, device):
    
                    # feature extractor to make image transforms same as pre-trained model
                    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
                    
                    # process the image according to vit and return tensor
                    def process_image(img):
                        img = Image.open(img).convert('RGB')
                        pixels = feature_extractor(img, return_tensors='pt')['pixel_values']
                        return pixels

                    # process image and send to device
                    img = process_image(img)
                    img = img.to(device)

                    # label dicts
                    target_to_label_2 = {'normal' : 0, 'abnormal': 1}
                    label_to_target_2 = {0 : 'normal', 1 : 'abnormal'}

                    # create ViT model for predictions
                    ViT_test_model = ViTForImageClassification.from_pretrained(
                    model_path,
                    num_labels=2,
                    id2label=label_to_target_2,
                    label2id=target_to_label_2
                    )

                    # move model to device
                    ViT_test_model.to(device)

                    # do a forward pass and get logits
                    with torch.no_grad():
                        outputs = ViT_test_model(img)
                        logits = outputs.logits

                    # get prediction and translate id2label
                    prediction = logits.argmax(-1)
                    label = ViT_test_model.config.id2label[prediction.item()]

                    return label

              
              elbow_pred_abnorm = predict_class(image, 'Elbow', device)  

              prediction = str(elbow_pred_abnorm)
              
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
