### Quick Walkthrough of the Pipeline
Use [this](LogoDetector/pipeline_demo.ipynb) colab notebook for a quick walkthrough of the pipeline

### Training a YOLO detection model
If you would like to train a YOLO model, you can do so by using the **train.py** file. This file is sourced from a GitHub repo by Ultralytics - a commonly used resource to train YOLO models. Details about this can be found at their [repo](https://github.com/ultralytics/yolov3). A basic command line argument to train the model is as follows.
```
$ python train.py --data coco.yaml --cfg yolov3.yaml --weights '' --batch-size 24

```
The following is the complete set of optional arguments that can be provided to train your own custom YOLO models.

```
Optional arguments for train.py:
  -h, --help            show this help message and exit
  --weights WEIGHTS     initial weights path
  --cfg CFG             model.yaml path
  --data DATA           data.yaml path
  --hyp HYP             hyperparameters path
  --epochs EPOCHS
  --batch-size BATCH_SIZE
                        total batch size for all GPUs
  --img-size IMG_SIZE [IMG_SIZE ...]
                        [train, test] image sizes
  --rect                rectangular training
  --resume [RESUME]     resume most recent training
  --nosave              only save final checkpoint
  --notest              only test final epoch
  --noautoanchor        disable autoanchor check
  --evolve              evolve hyperparameters
  --bucket BUCKET       gsutil bucket
  --cache-images        cache images for faster training
  --image-weights       use weighted image selection for training
  --device DEVICE       cuda device, i.e. 0 or 0,1,2,3 or cpu
  --multi-scale         vary img-size +/- 50%
  --single-cls          train multi-class data as single-class
  --adam                use torch.optim.Adam() optimizer
  --sync-bn             use SyncBatchNorm, only available in DDP mode
  --local_rank LOCAL_RANK
                        DDP parameter, do not modify
  --log-imgs LOG_IMGS   number of images for W&B logging, max 100
  --log-artifacts       log artifacts, i.e. final trained model
  --workers WORKERS     maximum number of dataloader workers
  --project PROJECT     save to project/name
  --name NAME           save to project/name
  --exist-ok            existing project/name ok, do not increment
  --quad                quad dataloader
```

### Testing YOLO model
If you wish to test your YOLO model on some data and want to see a summary of the performance, use the **test.py** file. These file is also sourced from a GitHub repo by Ultralytics - a commonly used resource to train YOLO models. Details about this can be found at their [repo](https://github.com/ultralytics/yolov3). A basic command line argument to train the model is as follows.
```
$ python test.py --data coco_test.yaml --weights '' --batch-size 24

```
The following is the complete set of optional arguments for testing your YOLO model.

```
optional arguments for test.py:
  -h, --help            show this help message and exit
  --weights WEIGHTS [WEIGHTS ...]
                        model.pt path(s)
  --data DATA           *.data path
  --batch-size BATCH_SIZE
                        size of each image batch
  --img-size IMG_SIZE   inference size (pixels)
  --conf-thres CONF_THRES
                        object confidence threshold
  --iou-thres IOU_THRES
                        IOU threshold for NMS
  --task TASK           'val', 'test', 'study'
  --device DEVICE       cuda device, i.e. 0 or 0,1,2,3 or cpu
  --single-cls          treat as single-class dataset
  --augment             augmented inference
  --verbose             report mAP by class
  --save-txt            save results to *.txt
  --save-hybrid         save label+prediction hybrid results to *.txt
  --save-conf           save confidences in --save-txt labels
  --save-json           save a cocoapi-compatible JSON results file
  --project PROJECT     save to project/name
  --name NAME           save to project/name
  --exist-ok            existing project/name ok, do not increment
  ```
### The Main file

If you'd like to detect, classify, implement open-set learning and even create ensembles of detectors then you need to use **detect.py**. This is a 
one-stop shop to implment our whole project once you have the weights for the trained detectors (YOLO or Faster RCNN). Please not that you may only 
two detection models using our code. This can be a Faster RCNN and YOLO, two Faster RCNN models or two YOLO models. The following is a simple
command line argument using detect.py.

```
$python detect.py --source ../proofpoint_data/images/demo -d rcnn -dw weights/rcnn.pth -dt 0.767 -c efficientnet_b3 -cw weights/efficientnet_b3.pt \
  --name demo --T config/Ts.p --train-avs config/activations/full/train/train_avs.pt --train-labs config/activations/full/train/train_labs.pt \
  --brand-mapping config/brand_mapping.p --open-type 2 --n-classes 1098 --save-img --save-json
```
The following is the complete set of optional arguments for detect.py.

```
optional arguments for detect.py:
  -h, --help            show this help message and exit
  -d DETECTOR, --detector DETECTOR
                        options: yolo (YOLOv3-SPP) or rcnn (Faster-RCNN) defaults to yolo
  -dw DETECTOR_WEIGHTS, --detector-weights DETECTOR_WEIGHTS
                        model.pt path(s)
  -dt DETECT_THRES, --detect-thres DETECT_THRES
                        object confidence threshold
  -c CLASSIFIER, --classifier CLASSIFIER
                        any model in timm library with associated --transform-sz
  -cw CLASSIFIER_WEIGHTS, --classifier-weights CLASSIFIER_WEIGHTS
                        model.pt path(s)
  --transform-sz TRANSFORM_SZ
                        data transform size of images for associated timm model in --classifier. Defaults to 300
  --rcnn-arch RCNN_ARCH
                        Faster-RCNN model architecture
  --source SOURCE       source
  --img-size IMG_SIZE   inference size (pixels)
  --iou-thres IOU_THRES
                        IOU threshold for NMS
  --device DEVICE       cuda device, i.e. 0 or 0,1,2,3 or cpu
  --save-json           save results as JSON to <name>/labels/predictions.json
  --save-img            save results as boxes on images
  --classes CLASSES [CLASSES ...]
                        filter by class: --class 0, or --class 0 2 3
  --agnostic-nms        class-agnostic NMS
  --augment             augmented inference
  --project PROJECT     save results to project/name
  --name NAME           save results to project/name
  --exist-ok            existing project/name ok, do not increment
  --train-avs TRAIN_AVS
                        path to activation vectors for train data
  --train-labs TRAIN_LABS
                        path to labels associated with activation vectors for train data
  --brand-mapping BRAND_MAPPING
                        path to brand mapping
  --n-classes N_CLASSES
                        number of classes model trained on
  --T T                 path to thresholds for open set learning (see open set learning docs)
  --open-type OPEN_TYPE
                        1 if last layer of classifier is activation vector, 2 if second last, 0 if concatination of last and second last
```
### Error Analysis
The **error_analysis.py** file can be used to perform error analysis on the detector performance. 

You may get an eloborate list of the arguments for thsi file by running the following command line code.

```
python error_analysis.py -h
```

### Perparing Open Set learning

The file **prepare_openset.py** prepares the files and folders necessary for open set classification and/or testing of open set performance. It is also used whenever we want to append classes to our existing lookup table.

You may get an eloborate list of the arguments for thsi file by running the following command line code.

```
python prepare_openset.py -h
```
