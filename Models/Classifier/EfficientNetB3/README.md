# Overview

This readme presents an overview of the `train_classifier.py` and `test_classifier.py` files. These files will allow you to set directories for train, validation, and test data, as well as set a few hyperparameters for the EfficientNet B3 classifier. In addition, you can set directories to store model weights and metric reports. 

**Note**: The default directories have been set according to how the data was stored in Google Drive/colab. Please ensure that you adjust the directories accordingly depending on how you are storing/retrieving the data and other files. 

## Train Classifier
To train the classifier, you can execute the `train.py` file in the command line. A sample command line argument is shown as follows:

```
python train_classifier.py --train_samples 35 --epochs 10
```

In the above argument, only classes that have more than 35 images in their training data are selected for training, and the model will train for 10 epochs. 

The following represent the full list of arguments that can be added along with descriptions of what each argument does. 

```
optional arguments:
  -h, --help: show this help message and exit
  --train_data TRAIN_DATA:  directory of training data
  --val_data VAL_DATA: directory of validation data
  --train_samples TRAIN_SAMPLES: minimum number of images required for class to be considered for training
  --epochs EPOCHS: number of epochs required to train model
  --learning_rate LEARNING_RATE: learning rate for EfficientNet B3
  --decay DECAY: learning rate decay during training
  --decay_epochs DECAY_EPOCHS: epoch to initialize learning rate decay (for example, if set to 15, decay will occur every 15 epochs)
  --checkpoint_path CHECKPOINT_PATH: directory to store best model weights
  --class_list CLASS_LIST: text file listing classes used during training; stored by default in the checkpoint path folder
  --weight_file WEIGHT_FILE: filename of model weights stored in checkpoint path
```

When the training file is run, the train and val data will be loaded and transformed accordingly. Then, depending on the value for *train_samples*, only classes that have the specified minimum number of samples will be used in training. A text file containing these classes will be generated, which is important because in open-set learning through [prepare_openset.py](LogoDetector/prepare_openset.py), you will need to isolate the data for these classes. This may require copying the relevant data entirely to a new directory, and having this class list will make this process more straightforward.  

Weights for the best performing model on the validation set will be saved. There will be print statements that indicate when data as loaded and as the training process is occurring. 

## Test Classifer
To test the classifier, you can execute the `test_classifer.py` file in the command line. A sample command line argument is shown as follows:

```
python test_classifier.py --train_samples 35
```

In the above argument, the testing is done on the classes who had more than 35 images in their training data. It is important to have this parameter in order to match the training and test data accordingly.

The following represent the full list of arguments that can be added along with descriptions of what each argument does. 

```
optional arguments:
  -h, --help: show this help message and exit
  --train_data TRAIN_DATA: directory of training data
  --test_data TEST_DATA: directory of validation data
  --train_samples TRAIN_SAMPLES: minimum number of images required for class to be considered for training
  --checkpoint_path CHECKPOINT_PATH: directory to store best model weights
  --weight_file WEIGHT_FILE: filename of model weights stored in checkpoint path
  --brand_metrics BRAND_METRICS: spreadsheet that stores metrics for each brand in test set; stored in checkpoint_path folder
  --metrics_summary METRICS_SUMMARY: spreadsheet that stores overall metrics for entire test set; stored in checkpoint_path folder
```

When this script is run, the training and test data will be loaded and transformed accordingly. The minimum number of samples parameter ensures that the index/target of the train and test data match so that accurate predictions are generated. The script will find the test classes also present in the training data, match the indices accordingly, and then use the model weights to load a new model and evaluate on the test data. The relevant metrics will be stored in spreadsheets for a more thorough summary of results. 
