import argparse
import time
from pathlib import Path

# ultralytics imports
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random, pad
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
import json

# classifier imports
from torch import nn
import timm
from torchvision import transforms
import pickle
from PIL import Image
from utils.classification import load_classifier, classifier_transforms, classify
import torchvision

# detectron 2 imports
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo


def check_assertions() -> None:
    """Check for all errors in command line inputs"""
    assert opt.save_img or opt.save_json == True, "At least one of --save-json and --save-img must be provided to output results"
    assert len(opt.detector_weights) <= 2, "Only a maximum of 2 models can be ensembled"
    for mod in opt.detector:
        assert mod in ["yolo", "rcnn"], "Options for --detector are 'yolo', or 'rcnn'"
    assert len(opt.detector_weights) == len(
        opt.detector), "--detector-weights and --detector must have same count"
    assert opt.classifier in [
        "efficientnet_b0", "efficientnet_b3"], "Options for --classifier are 'efficientnet_b0' or 'efficientnet_b3'"
    assert (opt.T is not None) and (opt.train_avs is not None) and (
        opt.train_labs is not None), "Options for --T, --train-avs, and --train-labs must be specified for open set detection"
    assert opt.open_type in [0, 1, 2], "Options for --open-type are 0, 1, and 2"


def detect(save_img: bool = False) -> None:
    """
    Main function for detection and classification. Calls upon yolo and rcnn detection
    and classification functions to do logo detection and open set classification
    """
    detector_type, detector_weights = opt.detector, opt.detector_weights
    check_assertions()

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name,
                                   exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if opt.save_json else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize: choose which models to use
    set_logging()
    device = select_device(opt.device)
    if len(detector_type) == 1:
        # if not ensembling choose between yolo and rcnn
        if detector_type[0] == "yolo":
            detector, imgsz = yolo_model(detector_weights[0], device)
        elif detector_type[0] == "rcnn":
            detector, imgsz = rcnn_model(detector_weights[0], float(opt.detect_thres[0]))
    else:
        # ensemble
        detectors, imgsz = ensemble_model(detector_type, detector_weights, device)

    # Brand mapping
    with open(opt.brand_mapping, "rb") as f:
        logos = pickle.load(f)

    # Load classifier
    classifier = load_classifier(opt.classifier, opt.classifier_weights,
                                 device, opt.n_classes, opt.open_type)

    # Load open set learning files
    train_labs = torch.load(opt.train_labs, map_location=torch.device(device))
    train_avs = torch.load(opt.train_avs, map_location=torch.device(device))
    with open(opt.T, "rb") as f:
        T = pickle.load(f)

    # Set Dataloader
    dataset = LoadImages(opt.source, img_size=imgsz)

    # Run inference
    t0 = time.time()
    if len(detector_type) == 1:
        # detect and classify using one of yolo or rcnn
        if detector_type[0] == "yolo":
            detect_yolo(detector, classifier, T, train_avs, train_labs,
                        imgsz, dataset, logos, device, save_dir)
        elif detector_type[0] == "rcnn":
            detect_rcnn(detector, classifier, T, train_avs, train_labs,
                        imgsz, dataset, logos, device, save_dir)
    else:
        # detect and classify using ensemble
        detect_ensemble(detectors, classifier, T, train_avs, train_labs,
                        imgsz, dataset, logos, device, save_dir)

    # total detection and classification time
    print(f'Done. ({time.time() - t0:.3f}s)')


def yolo_model(weights: str, device: str) -> tuple:
    """Return YOLO model and associated image size"""
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load([weights], map_location=device)  # load FP32 model
    imgsz = check_img_size(opt.img_size, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16
    return model, imgsz


def rcnn_model(weights: str, detect_thres: float) -> tuple:
    """Return RCNN model and associated image size"""
    cfg = get_cfg()  # init config
    cfg.merge_from_file(model_zoo.get_config_file(opt.rcnn_arch))  # read rcnn architecture
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # num classes
    cfg.MODEL.WEIGHTS = weights  # path to the model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = detect_thres  # set a custom detection threshold
    model = DefaultPredictor(cfg)  # load model
    imgsz = check_img_size(opt.img_size)  # check img_size
    return model, imgsz


def ensemble_model(detector_types: list, detector_weights: list, device: str) -> tuple:
    """Return Ensemble model and associated image size"""
    models = []  # list of models
    for i in range(2):
        # can have maximum of two models in ensemble
        if detector_types[i] == 'yolo':
            # if yolo append yolo model
            model, imgsz = yolo_model(detector_weights[i], device)
            models.append(model)
        elif detector_types[i] == 'rcnn':
            # if rcnn append yolo model
            detect_thres = float(opt.detect_thres[i])
            model, imgsz = rcnn_model(detector_weights[i], detect_thres)
            models.append(model)
    return models, imgsz


def box_area(b: torch.Tensor) -> torch.Tensor:
    """Returns area of input box"""
    return (b[2] - b[0]) * (b[3] - b[1])


def iou(bb1, bb2):
    """Returns intersection over union (iou) between input bounding boxes"""
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    # if no intersection, return 0
    if x_right < x_left or y_bottom < y_top:
        return 0

    # intersection area
    w = (x_right - x_left)
    h = (y_bottom - y_top)
    intersection_area = w * h

    # compute the area of both AABBs
    bb1_area = box_area(bb1)
    bb2_area = box_area(bb2)

    # iou
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou


def classify_box(classifier: nn.Sequential, T: dict, train_avs: torch.Tensor, train_labs: torch.Tensor,
                 device: str, transform: torchvision.transforms, xyxy: torch.Tensor, im0: torch.Tensor,
                 logos: dict) -> tuple:
    """Use classifer to do open set prediction on input bounding box xyxy."""
    x1, y1, x2, y2 = xyxy  # bounding box coordinates
    sub = im0[int(y1.item()):int(y2.item()), int(x1.item()):int(x2.item())]  # extracted logo matrix

    if (sub.shape[0] == 0) or (sub.shape[1] == 0):
        # return of logo being just a line
        return "Unknown", 0, torch.Tensor([0.0]).to(device)

    sub = cv2.cvtColor(sub, cv2.COLOR_BGR2RGB)  # BGR image to RGB

    # apply transformations
    trans_img = transform(sub).to(device)
    trans_img = trans_img.view(1, 3, opt.transform_sz, opt.transform_sz)

    # classifier forward pass to get activation vector
    av = classify(classifier, opt.open_type, trans_img)

    ### open set classification ###
    dist = torch.norm(train_avs - av, dim=1, p=None)  # distances to all lookup table AVs
    knn = dist.topk(1, largest=False)  # get closest AV
    distance = knn.values  # distance to closest AV
    nearest_label = train_labs[knn.indices].item()  # label of closest AV
    if distance < T[nearest_label]:
        # label is nearest label if with distance within threshold
        label = logos[nearest_label]
    else:
        label = 'Unknown'
    return label, nearest_label, distance


def jsonify_logo(xyxy: torch.Tensor, score: torch.Tensor, label: str, nearest_label: str,
                 distance: torch.Tensor, stem: Path.stem, jlist: list) -> None:
    """Append bounding box results to json results list"""
    jlist.append({'image_id': str(stem),
                  'bbox': [int(c.item()) for c in xyxy],
                  'score': score.item(),
                  'label': label,
                  'nearest_label': nearest_label,
                  'distance': distance.item()})


def write_json(save_dir: Path, jlist: list) -> None:
    """Write out json results list using pickle"""
    pred_json = str(save_dir / "labels/predictions.json")
    with open(pred_json, 'w') as f:
        json.dump(jlist, f)


def process_paths(path: str, dataset: torch.utils.data.DataLoader, save_dir: Path) -> tuple:
    """Return path to save image and text results"""
    p, frame = path, getattr(dataset, 'frame', 0)
    p = Path(p)  # to Path
    save_path = str(save_dir / p.name)  # img.jpg
    return save_path, p.stem

def ensemble(boxes: tuple, confs: tuple, labels: tuple, thres: float):
    """
    Ensembling function. Accept all boxes from first element of input boxes and any
    additional boxes from second element of input boxes.

    thres: threshold for iou of boxes for being considered a different box
    """
    boxes1, boxes2 = boxes
    labels1, labels2 = labels
    confs1, confs2, = confs

    # default outputs if one or both models found no boxes
    if (boxes1.shape[0] == 0) and (boxes2.shape[0] == 0):
        return boxes1, confs1, labels1
    elif boxes1.shape[0] == 0:
        return boxes2, confs2, labels2

    # initialize ensembled boxes
    new_boxes = boxes1.clone()
    new_labels = labels1.copy()
    new_confs = confs1.clone()

    # loop over all boxes in boxes2 to find new boxes
    for i in range(boxes2.shape[0]):
        box = boxes2[i]
        lab = labels2[i]
        conf = confs2[i].reshape(1)
        add = True
        for j in range(boxes1.shape[0]):
            done = boxes1[j]
            iou_boxes = iou(box, done)
            if iou_boxes > thres:
                add = False
                if lab[0] == 'Unknown':
                    new_labels[j] = lab
                break
        if add == True:
            new_boxes = torch.vstack((new_boxes, box.reshape(1, -1)))
            new_confs = torch.cat((new_confs, conf))
            new_labels.append(lab)
    return new_boxes, new_confs, new_labels


def yolo_predictions(img: torch.Tensor, half: torch.tensor, detector: nn.Sequential,
                     detect_thres: float, classifier: nn.Sequential, T: dict, train_avs: torch.Tensor,
                     train_labs: torch.Tensor, device: str, transform: torchvision.transforms, im0: torch.Tensor,
                     logos: dict) -> tuple:
    """
    Classify all boxes in image for YOLO predictions.
    """
    # initializations
    boxes = torch.Tensor([])
    labs = []

    # image transformation to yolo format
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = detector(img, augment=opt.augment)[0]
    # Apply NMS
    pred = non_max_suppression(pred, detect_thres, opt.iou_thres,
                               classes=opt.classes, agnostic=opt.agnostic_nms)

    for i, det in enumerate(pred):
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # init result collections
            boxes = []
            confs = []
            # loop over boxes and classify
            for *xyxy, conf, cls in reversed(det):
                label, nearest_label, distance = classify_box(
                    classifier, T, train_avs, train_labs, device, transform, xyxy, im0, logos)
                # save results
                boxes.append(xyxy)
                confs.append(conf)
                labs.append((label, nearest_label, distance))
            boxes = torch.Tensor(boxes).to(device)
            confs = torch.Tensor(confs).to(device)
    if len(boxes) == 0:
        # case when no boxes found
        boxes = torch.Tensor([]).to(device)
        confs = torch.Tensor([]).to(device)
    return boxes, confs, labs


def rcnn_predictions(detector: nn.Sequential, classifier: nn.Sequential, T: dict, train_avs: torch.Tensor,
                     train_labs: torch.Tensor, device: str, transform: torchvision.transforms,
                     im0: torch.Tensor, logos: dict) -> tuple:
    """
    Classify all boxes in image for RCNN predictions.
    """
    # Inference + NMS
    pred = detector(im0)['instances'].get_fields()
    # init or get result collections
    boxes = pred['pred_boxes'].tensor
    confs = pred['scores']
    labs = []
    # loop over boxes and classify
    for i in range(confs.shape[0]):
        conf = confs[i]
        xyxy = boxes[i]
        label, nearest_label, distance = classify_box(
            classifier, T, train_avs, train_labs, device, transform, xyxy, im0, logos)
        labs.append((label, nearest_label, distance))
    return boxes, confs, labs


def build_results(boxes: torch.Tensor, labs: torch.Tensor, logos: dict, implot: torch.Tensor,
                  colors: list, confs:  torch.Tensor, stem: Path.stem, save_path: str, jlist: list) -> None:
    """
    Save results to json list and/or to image. If opt.save_img, also save a copy of the image.
    """
    # loop over boxes and save them to json or image
    for i in range(boxes.shape[0]):
        xyxy = boxes[i]
        label, nearest_label, distance = labs[i]
        if opt.save_img:
            plot_one_box(xyxy, implot, label=label,
                         color=colors[int(nearest_label)], line_thickness=3)
        if opt.save_json:
            jsonify_logo(xyxy, confs[i], label, logos[nearest_label], distance, stem, jlist)

    # write image
    if opt.save_img:
        cv2.imwrite(save_path, implot)


def detect_ensemble(detectors: list, classifier: nn.Sequential, T: dict, train_avs: torch.Tensor,
                    train_labs: torch.Tensor, imgsz: int, dataset: torch.utils.data.DataLoader,
                    logos: dict, device: str, save_dir: str) -> None:
    """
    Ensemble detector models and classify their collectively chosen boxes.
    """
    transform = classifier_transforms(opt.transform_sz)  # transforms
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in list(logos.keys())]  # get colors for plotting
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    for i in range(len(opt.detector)):
        if opt.detector[i] == 'yolo':  # yolo inits
            half = device.type != 'cpu'
            _ = detectors[i](img.half() if half else img) if device.type != 'cpu' else None
    jlist = []  # init write json
    implot = None  # init write images
    for path, img, im0, vid_cap in dataset:
        save_path, stem = process_paths(path, dataset, save_dir)  # save paths
        if opt.save_img:
            implot = pad(im0, ((100, 100), (0, 0), (0, 0)), mode='constant',
                         constant_values=169)  # init image for plotting
        # init results collections
        preds = []
        labels = []
        confidences = []
        # loop over detectors to get their results
        for i in range(len(opt.detector)):
            # yolo
            if opt.detector[i] == 'yolo':
                boxes, confs, labs = yolo_predictions(img, half, detectors[i], float(
                    opt.detect_thres[i]), classifier, T, train_avs, train_labs, device, transform, im0, logos)
            else:
                # faster-rcnn
                boxes, confs, labs = rcnn_predictions(
                    detectors[i], classifier, T, train_avs, train_labs, device, transform, im0, logos)
            preds.append(boxes)
            labels.append(labs)
            confidences.append(confs)

        boxes, confs, labs = ensemble(tuple(preds), tuple(confidences), tuple(labels), 0.4)  # hardcoded threshold
        build_results(boxes, labs, logos, implot, colors, confs, stem, save_path, jlist)  # AR filler confidence here

    if opt.save_json:  # write json results
        write_json(save_dir, jlist)

    print(f"Results saved to {save_dir}")


def detect_yolo(detector: nn.Sequential, classifier: nn.Sequential, T: dict, train_avs: torch.Tensor,
                train_labs: torch.Tensor, imgsz: int, dataset: torch.utils.data.DataLoader, logos: dict,
                device: str, save_dir: str) -> None:
    """
    Detect using YOLO and classify resulting boxes.
    """
    transform = classifier_transforms(opt.transform_sz)  # transforms
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in list(logos.keys())]  # get colors
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # yolo inits
    half = device.type != 'cpu'
    _ = detector(img.half() if half else img) if device.type != 'cpu' else None
    jlist = []  # write json
    implot = None  # write images
    for path, img, im0, vid_cap in dataset:
        save_path, stem = process_paths(path, dataset, save_dir)
        if opt.save_img:
            implot = pad(im0, ((100, 100), (0, 0), (0, 0)), mode='constant',
                         constant_values=169)  # init image for plotting

        # get predcitions for image
        boxes, confs, labs = yolo_predictions(img, half, detector, float(opt.detect_thres[0]), classifier,
                                              T, train_avs, train_labs, device, transform, im0, logos)
        build_results(boxes, labs, logos, implot, colors, confs, stem, save_path, jlist)
        

    if opt.save_json:  # write json results
        write_json(save_dir, jlist)

    print(f"Results saved to {save_dir}")


def detect_rcnn(detector: nn.Sequential, classifier: nn.Sequential, T: dict, train_avs: torch.Tensor,
                train_labs: torch.Tensor, imgsz: int, dataset: torch.utils.data.DataLoader, logos: dict,
                device: str, save_dir: str) -> None:
    """
    Detect using Faster-RCNN and classify resulting boxes.
    """
    transform = classifier_transforms(opt.transform_sz)  # transforms
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in list(logos.keys())]  # get colors
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    jlist = []  # write json
    implot = None  # write images
    for path, img, im0, vid_cap in dataset:
        save_path, stem = process_paths(path, dataset, save_dir)
        if opt.save_img:
            implot = pad(im0, ((100, 100), (0, 0), (0, 0)), mode='constant',
                         constant_values=169)  # init image for plotting

        # get predcitions for image
        boxes, confs, labs = rcnn_predictions(
            detector, classifier, T, train_avs, train_labs, device, transform, im0, logos)
        build_results(boxes, labs, logos, implot, colors, confs, stem, save_path, jlist)

    if opt.save_json:  # write json results
        write_json(save_dir, jlist)
    print(f"Results saved to {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--detector', action='append',
                        help='options: yolo (YOLOv3-SPP) or rcnn (Faster-RCNN) defaults to yolo')
    parser.add_argument('-dw', '--detector-weights', action='append', help='model.pt path(s)')
    parser.add_argument('-dt', '--detect-thres', action='append',
                        help='object confidence threshold')
    parser.add_argument('-c', '--classifier', type=str, default='efficientnet_b3',
                        help='any model in timm library with associated --transform-sz')
    parser.add_argument('-cw', '--classifier-weights', type=str,
                        default='weights/efficientnet_b3.pt', help='model.pt path(s)')
    parser.add_argument('--transform-sz', type=int, default=300,
                        help='data transform size of images for associated timm model in --classifier. Defaults to 300')
    parser.add_argument('--rcnn-arch', type=str, default='COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml',
                        help='Faster-RCNN model architecture')
    parser.add_argument('--source', type=str, default='data/images',
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=2560, help='inference size (pixels)')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-json', action='store_true',
                        help='save results as JSON to <name>/labels/predictions.json')
    parser.add_argument('--save-img', action='store_true', help='save results as boxes on images')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--train-avs', default=None, type=str,
                        help='path to activation vectors for train data')
    parser.add_argument('--train-labs', default=None, type=str,
                        help='path to labels associated with activation vectors for train data')
    parser.add_argument('--brand-mapping', default="config/brand_mapping.p",
                        type=str, help='path to brand mapping')
    parser.add_argument('--n-classes', default=32, type=int,
                        help='number of classes model trained on')
    parser.add_argument('--T', default=None, type=str,
                        help='path to thresholds for open set learning (see open set learning docs)')
    parser.add_argument('--open-type', type=int, default=2,
                        help='1 if last layer of classifier is activation vector, 2 if second last, 0 if concatination of last and second last; defaults to 2')
    opt = parser.parse_args()
    print(opt)
    check_requirements()

    with torch.no_grad():
        detect()
