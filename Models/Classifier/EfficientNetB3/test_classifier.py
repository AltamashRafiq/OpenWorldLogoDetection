##### Import libraries #####
import os
import random
import shutil

import glob
import piexif
import cv2
from google.colab.patches import cv2_imshow
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from torchvision.utils import make_grid 
from torchvision.utils import save_image
import torchvision.transforms as transforms

from sklearn import metrics

from tqdm import tqdm
import argparse

##### Image Preprocessing #####
# Create class that resizes and pads (reference: https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/)
class resize_and_pad:
  def __init__(self, final_size = 300):
    self.final_size = final_size

  def __call__(self, image):
    # Obtain image size
    original_size = image.size  # original_size[0] is in (width, height) format

    # Obtain aspect ratio
    ratio = float(self.final_size)/max(original_size)
    new_size = tuple([int(x*ratio) for x in original_size])

    # Resize image based on largest dimension 
    image = image.resize(new_size, Image.ANTIALIAS)
    
    # Create a new image and paste the resized on it
    new_im = Image.new("RGB", (self.final_size, self.final_size), color = (255,255,255))
    new_im.paste(image, ((self.final_size-new_size[0])//2,
                        (self.final_size-new_size[1])//2))
    
    return new_im
    

# Create transformations
transform_train  = transforms.Compose([
        resize_and_pad(final_size=300), 
        transforms.RandomResizedCrop(300, scale = (0.5,1)),
        transforms.RandomRotation(degrees = 90),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
        resize_and_pad(final_size=300),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

##### Load and transform data #####
def load_train_data(train_dir):
    train_set = datasets.ImageFolder(root = train_dir, transform = transform_train)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size = 8, shuffle = True, num_workers = 2)
    
    return train_set, trainloader
    
def load_test_data(test_dir):
    test_set = datasets.ImageFolder(root = test_dir, transform = transform_test)
    testloader = torch.utils.data.DataLoader(test_set, batch_size = 32, shuffle = False, num_workers = 2)
    
    return test_set, testloader
    
    
def specify_classes(train_dir, train_set, num_samples):
  # Obtain brand index list
  brand_idx_list = [] 
  for brand in os.listdir(train_dir):
    if len(os.listdir(os.path.join(train_dir, brand))) > num_samples:
      brand_idx_list.append(train_set.class_to_idx[brand])
  
  return brand_idx_list
  
def metrics_summary(true_class, pred_class):
    # Obtain accuracy, precision, recall, f1-score on predicted labels
    accuracy = metrics.accuracy_score(true_class, pred_class)
    precision = metrics.precision_score(true_class, pred_class, average = 'weighted', zero_division='warn')
    recall = metrics.recall_score(true_class, pred_class, average = 'weighted')
    f1_score = metrics.f1_score(true_class, pred_class, average = 'weighted')
    
    print(f'Accuracy: {accuracy*100:3f}\n'
          f'Precision: {precision*100:3f}\n'
          f'Recall: {recall*100:3f}\n'
          f'F1-Score: {f1_score*100:3f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Evaluate trained B3 classifier on test data and save results')
    parser.add_argument('--train_data', type = str, default = 'train_with_extracted', help = 'directory of training data')
    parser.add_argument('--test_data', type = str, default = 'test_with_extracted', help = 'directory of validation data')
    parser.add_argument('--train_samples', type = int, default = 1, help = 'minimum number of images required for class to be considered for training')
    parser.add_argument('--checkpoint_path', type = str, default = 'model_outputs/EfficientNetB3/Intermediate_Model', help = 'directory to store best model weights')
    parser.add_argument('--weight_file', type = str, default = 'model.pt', help = 'filename of model weights stored in checkpoint path')
    parser.add_argument('--brand_metrics', type = str, default = 'test_set_report.csv', help = 'spreadsheet that stores metrics for each brand in test set; stored in checkpoint_path folder')
    parser.add_argument('--metrics_summary', type = str, default = 'test_set_metric_report.csv', help = 'spreadsheet that stores overall metrics for entire test set; stored in checkpoint_path folder')
    args = parser.parse_args()
    
    # Load training and validation data
    train_set, trainloader = load_train_data(args.train_data)
    print('Train data loaded')
    
    # Generate original train (brand) and test dictionaries
    brand_dict = train_set.class_to_idx
    brand_dict = {v:k for k, v in brand_dict.items()}
    
    # Obtain brands/indices with specified number of samples
    brand_idx_list = specify_classes(train_dir = args.train_data, train_set = train_set, num_samples = args.train_samples)
    
    # Replace train_set samples
    train_set.samples = [(s[0],s[1]) for s in train_set.samples if s[1] in brand_idx_list]
    
    # Remap indices to match targets between original and subsetted data (restart indices at 0)
    indices = np.unique(np.array([s[1] for s in train_set.samples]))
    index_map = {v:k for v,k in enumerate(indices)}
    index_remap = {k:v for v, k in enumerate(indices)}
    
    # Load test data
    test_set, testloader = load_test_data(args.test_data)
    print('Test data loaded')
    
    # Obtain test set dictionary
    test_dict = test_set.class_to_idx
    test_dict = {v:k for k, v in test_dict.items()}
    
    # Obtain train class list
    train_classes = [brand_dict[idx] for idx in brand_idx_list]
    
    # Find classes in both train and test
    test_classes = [brand for brand in set(test_set.classes).intersection(train_classes)]
    
    # Update test samples
    test_set.samples = [(s[0], s[1]) for s in test_set.samples if test_dict[s[1]] in test_classes]
    
    # Match train and test indices 
    test_to_train = dict(zip([test_set.class_to_idx[brand] for brand in test_classes],
                             [train_set.class_to_idx[brand] for brand in test_classes]))
                             
    
    test_set.samples = [(s[0], index_remap[test_to_train[s[1]]]) for s in test_set.samples]
    
    # Obtain all true labels on validation or test set
    true_class = [s[1] for s in test_set.samples]
    print(true_class)
    
    # Create empty list for predictions
    pred_class = []
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = timm.create_model('efficientnet_b3', pretrained = False)
    model.classifier = torch.nn.Linear(model.classifier.in_features, len((brand_idx_list)))
    model.to(device)
    
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, args.weight_file))['m'])
    model.eval()
    
    # Get predicted labels on all validation data
    with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(testloader):
          #target_list.append(targets)
          # Copy inputs to device
          inputs = inputs.to(device)
          # Generate output from the model
          outputs = model(inputs)
          # Calculate predicted labels
          _, predicted = outputs.max(1)
          pred_class.append(predicted.cpu().numpy())
    print('Prediction completed')
    
    # Flatten the list of arrays
    pred_class = np.concatenate(pred_class).ravel().tolist()
    
    # Generate predictions
    metrics_summary(true_class, pred_class)
    
    # Create class labels (replace index with actual brand name)
    true_class_labels = true_class.copy()
    true_class_labels = [brand_dict[index_map[i]] for i in true_class_labels]
    
    pred_class_labels = pred_class.copy()
    pred_class_labels = [brand_dict[index_map[i]] for i in pred_class_labels]
        
    # Generate classification report
    report = metrics.classification_report(true_class_labels, pred_class_labels, digits = 4, output_dict = True)

    # Convert report to pdf
    report_df = pd.DataFrame(report).transpose()
    
    # Split dataframe 
    brand_report = report_df.iloc[:-3,:]
    metrics_report = report_df.iloc[-3:,:]

    brand_report = brand_report.sort_values(by = 'support', ascending = False).reset_index().rename(columns = {'index': 'brand'})
    
    # Save reports to csv
    brand_report.to_csv(os.path.join(args.checkpoint_path, args.brand_metrics), index = False)
    
    # Metrics report
    metrics_report.to_csv(os.path.join(args.checkpoint_path, args.metrics_summary))
    
    print('Reports saved')
