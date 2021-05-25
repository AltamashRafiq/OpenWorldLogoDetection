##### Import libraries #####
import os
import random
import shutil

import glob
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

transform_val = transforms.Compose([
        resize_and_pad(final_size=300), 
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])



# Load and transform train data
def load_train_data(train_dir):
    train_set = datasets.ImageFolder(root = train_dir, transform = transform_train)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size = 8, shuffle = True, num_workers = 2)
    
    return train_set, trainloader
    

# Load and transform validation data
def load_val_data(val_dir):
    val_set = datasets.ImageFolder(root = val_dir, transform = transform_val)
    valloader = torch.utils.data.DataLoader(val_set, batch_size = 32, shuffle = False, num_workers = 2)
    
    return val_set, valloader
    

# Obtain indices of brands that have more than the requested number of samples in training data
def specify_classes(train_dir, train_set, num_samples):
  # Initialize list to obtain indices of relevant brands
  brand_idx_list = [] 
  for brand in os.listdir(train_dir):
    if len(os.listdir(os.path.join(train_dir, brand))) > num_samples:
      brand_idx_list.append(train_set.class_to_idx[brand])
  
  return brand_idx_list

# Save list of classes into text file
def save_classes(directory, filename, lst):
  with open(os.path.join(directory, filename),'w') as file:
    for item in lst:
      file.write(f'{item}\n')

###### Train Model ######
def train_model(model, lr, epochs, decay_epochs, decay, checkpoint_path, weight_file):

    # Set up loss function 
    criterion = nn.CrossEntropyLoss()
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr = lr)

    # Start the training/validation process
    global_step = 0
    best_val_acc = 0
    
    # Copy learning rate
    current_learning_rate = lr
    
    # Initialize lists to store loss and accuracy values
    train_loss_list = np.zeros(epochs)
    train_acc_list = np.zeros(epochs)
    val_loss_list = np.zeros(epochs)
    val_acc_list = np.zeros(epochs)

    # Run model
    for i in tqdm(range(epochs)):
        # Switch to train mode
        model.train()
        print("Epoch %d:" %i)
    
        total_examples = 0
        correct_examples = 0
    
        train_loss = 0
        train_acc = 0
        
        # Train the training dataset for 1 epoch.
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # Copy inputs to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Zero the gradient
            optimizer.zero_grad()
            # Generate output
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Now backward loss
            loss.backward()
            # Apply gradient
            optimizer.step()
            # Calculate predicted labels
            _, predicted = outputs.max(1)
            total_examples += predicted.size(0)
            correct_examples += predicted.eq(targets).sum().item()
            train_loss += loss
            global_step += 1
                    
        avg_loss = train_loss / (batch_idx + 1)
        avg_acc = correct_examples / total_examples
        print("Training loss: %.4f, Training accuracy: %.4f" %(avg_loss, avg_acc))
        #print(datetime.datetime.now())
        # Validate on the validation dataset
        print("Validation...")
        total_examples = 0
        correct_examples = 0
    
        # Save values
        train_loss_list[i] = avg_loss
        train_acc_list[i] = avg_acc
        
        model.eval()
    
        val_loss = 0
        val_acc = 0
        
        # Disable gradient during validation
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                # Copy inputs to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                # Zero the gradient
                optimizer.zero_grad()
                # Generate output from the DNN.
                outputs = model(inputs)
                loss = criterion(outputs, targets)            
                # Calculate predicted labels
                _, predicted = outputs.max(1)
                total_examples += predicted.size(0)
                correct_examples += predicted.eq(targets).sum().item()
                val_loss += loss
    
        avg_loss = val_loss / len(valloader)
        avg_acc = correct_examples / total_examples
    
        # Save values
        val_loss_list[i] = avg_loss
        val_acc_list[i] = avg_acc
        
        print("Validation loss: %.4f, Validation accuracy: %.4f" % (avg_loss, avg_acc))
    
            # Handle the learning rate scheduler.
        if i % decay_epochs == 0 and i != 0:
            current_learning_rate = current_learning_rate * decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_learning_rate
            print("Current learning rate has decayed to %f" %current_learning_rate)
        
        # Save for checkpoint
        if avg_acc > best_val_acc:
            best_val_acc = avg_acc
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            print("Saving ...")
            state = {'m': model.state_dict(),
                     'epoch': i,
                     'lr': current_learning_rate}
            torch.save(state, os.path.join(checkpoint_path, weight_file))
        print()
    
    print("Optimization finished.")
    print(f'Best validation accuracy: {best_val_acc}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train EfficientNet B3 classifier on specified number of classes of logos')
    parser.add_argument('--train_data', type = str, default = 'train_with_extracted', help = 'directory of training data')
    parser.add_argument('--val_data', type = str, default = 'val_with_extracted', help = 'directory of validation data')
    parser.add_argument('--train_samples', type = int, default = 1, help = 'minimum number of images required for class to be considered for training')
    parser.add_argument('--epochs', type = int, default = 30, help = 'number of epochs required to train model')
    parser.add_argument('--learning_rate', type = float, default = 0.0001, help = 'learning rate for EfficientNet B3')
    parser.add_argument('--decay', type = int, default = 0.1, help = 'learning rate decay during training')
    parser.add_argument('--decay_epochs', type = int, default = 15, help = 'epoch to initialize learning rate decay')
    parser.add_argument('--checkpoint_path', type = str, default = 'model_outputs/EfficientNetB3/Intermediate_Model', help = 'directory to store best model weights')
    parser.add_argument('--class_list', type = str, default = 'classes.txt', help = 'text file listing classes used during training; stored by default in the checkpoint path folder')
    parser.add_argument('--weight_file', type = str, default = 'model.pt', help = 'filename of model weights stored in checkpoint path')
    args = parser.parse_args()
    
    # Load training and validation data
    train_set, trainloader = load_train_data(args.train_data)
    print('Train data loaded')
    val_set, valloader = load_val_data(args.val_data)
    print('Val data loaded')
    
    # Generate original train (brand) dictionary
    brand_dict = train_set.class_to_idx
    brand_dict = {v:k for k, v in brand_dict.items()}
    
    # Obtain brands/indices with specified number of samples
    brand_idx_list = specify_classes(train_dir = args.train_data, train_set = train_set, num_samples = args.train_samples)
    
    # Obtain and save specific classes used in training
    train_classes = [brand_dict[idx] for idx in brand_idx_list]
    save_classes(args.checkpoint_path, args.class_list, train_classes)
    print(f'Classes used in training: {train_classes}')
    
    # Replace train_set samples
    train_set.samples = [(s[0],s[1]) for s in train_set.samples if s[1] in brand_idx_list]
    val_set.samples = [(s[0],s[1]) for s in val_set.samples if s[1] in brand_idx_list]
    
    # Generate index mapping for new data (restart classes at index 0 for PyTorch)
    indices = np.unique(np.array([s[1] for s in train_set.samples]))
    
    # Map old index to new and vice versa
    index_map = {v:k for v,k in enumerate(indices)}
    index_remap = {k:v for v,k in enumerate(indices)}
    
    # Remap sample targets 
    train_set.samples = [(s[0], index_remap[s[1]]) for s in train_set.samples]
    val_set.samples = [(s[0], index_remap[s[1]]) for s in val_set.samples]
    
    #### Train model ####
    # Run on GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.get_device_name(0)

    # Load model from timm library
    m = timm.create_model('efficientnet_b3', pretrained = True)
    m.classifier = torch.nn.Linear(m.classifier.in_features, len(brand_idx_list))
    m.to(device)
    
    # Run training 
    train_model(model = m, 
                lr = args.learning_rate, 
                epochs = args.epochs, 
                decay_epochs = args.decay_epochs, 
                decay = args.decay, 
                checkpoint_path = args.checkpoint_path, 
                weight_file = args.weight_file)