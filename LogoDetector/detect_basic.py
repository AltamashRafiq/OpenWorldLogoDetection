import argparse
import time
from pathlib import Path

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

# detectron 2 imports
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

def detect(save_img = False):
    source, view_img, save_txt = opt.source, opt.view_img, opt.save_txt
    detector_type, detector_weights = opt.detector, opt.detector_weights
    
    assert detector_type in ["yolo", "rcnn"], "Options for --detector are 'yolo' or 'rcnn'"
    
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    # Initialize
    set_logging()
    device = select_device(opt.device)
    if detector_type == "yolo":
        detector, imgsz = yolo_model(detector_weights, device)
    elif detector_type == "rcnn":
        detector, imgsz = rcnn_model(detector_weights)
        
    # Set Dataloader
    dataset = LoadImages(source, img_size = imgsz)
    
    # Run inference
    t0 = time.time()
    
    if detector_type == "yolo":
        detect_yolo(detector, imgsz, dataset, device, save_dir, save_txt, save_img)
    elif detector_type == "rcnn":
        detect_rcnn(detector, imgsz, dataset, device, save_dir, save_txt, save_img)
        
    print(f'Done. ({time.time() - t0:.3f}s)')
    
def yolo_model(weights, device):
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load([weights], map_location=device)  # load FP32 model
    imgsz = check_img_size(opt.img_size, s = model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16
    return model, imgsz
    
def rcnn_model(weights):
    # Load model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(opt.rcnn_arch))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = weights  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = opt.detect_thres   # set a custom testing threshold
    model = DefaultPredictor(cfg)
    imgsz = check_img_size(opt.img_size)  # check img_size
    return model, imgsz
    
def make_int(a):
    return int(a.item() * opt.resize_ann)

def write_logo(xyxy, txt_path):
    x1, y1, x2, y2 = xyxy
    x1, y1, x2, y2 = make_int(x1), make_int(y1), make_int(x2), make_int(y2)
    line = f"{x1} {y1} {x2} {y2}\n"
    with open(txt_path + '.txt', 'a') as f:
        f.write(line)
                          
def jsonify_logo(xyxy, score, stem, jdict):
    jdict.append({'image_id': str(stem),
                  'bbox': [make_int(c) for c in xyxy],
                  'score': score.item()})
                  
def write_json(save_dir, jdict):
    pred_json = str(save_dir / f"predictions.json")
    with open(pred_json, 'w') as f:
        json.dump(jdict, f)
    
def process_paths(path, dataset, save_dir):
    p, frame = path, getattr(dataset, 'frame', 0)
    p = Path(p)  # to Path
    save_path = str(save_dir / p.name)  # img.jpg
    txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
    return save_path, txt_path, p.stem
    
def time_verbose(t1, t2, t3, t4):
    print(f' Done. D: {t2 - t1:.3f}s C: {t4 - t3:.3f}s')
    
def detect_yolo(detector, imgsz, dataset, device, save_dir, save_txt, save_img):
    # Run inference
    half = device.type != 'cpu'
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = detector(img.half() if half else img) if device.type != 'cpu' else None  # run once
    jdict = [] # write json
    for path, img, im0, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = detector(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.detect_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        
        for i, det in enumerate(pred):  # detections per image
            save_path, txt_path, stem = process_paths(path, dataset, save_dir)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                t3 = time.time()
                for *xyxy, conf, cls in reversed(det):
                    jsonify_logo(xyxy, conf, stem, jdict)
                    
                t4 = time.time()
            # Print time (inference + NMS)
            time_verbose(t1, t2, t3, t4)
    
    write_json(save_dir, jdict)  
    if save_txt or save_img:
        s = f'\n{len(list(save_dir.glob("labels/*.txt")))} labels saved to {save_dir / "labels"}' if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

def detect_rcnn(detector, imgsz, dataset, device, save_dir, save_txt, save_img):
    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    jdict = [] # write json
    for path, img, im0, vid_cap in dataset:
        # Inference
        t1 = time_synchronized()
        pred = detector(im0)['instances'].get_fields()
        t2 = time_synchronized()
        save_path, txt_path, stem = process_paths(path, dataset, save_dir)

        # Write results
        boxes = pred['pred_boxes'].tensor
        scores = pred['scores']
        t3 = time.time()
        for i in range(scores.shape[0]):
            jsonify_logo(boxes[i], scores[i], stem, jdict)
        t4 = time.time()
        # Print time (inference + NMS)
        time_verbose(t1, t2, t3, t4)

    write_json(save_dir, jdict)
    if save_txt or save_img:
        s = f'\n{len(list(save_dir.glob("labels/*.txt")))} labels saved to {save_dir / "labels"}' if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', type=str, default = 'yolo', help='options: yolo (YOLOv3-SPP) or rcnn (Faster-RCNN) defaults to yolo')
    parser.add_argument('--detector-weights', type=str, default='weights/yolov3.pt', help='model.pt path(s)')
    parser.add_argument('--rcnn-arch', type=str, default='COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml', help='Faster-RCNN model architecture')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=2560, help='inference size (pixels)')
    parser.add_argument('--detect-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--resize-ann', type=float, default=1, help='multiplier to resize annotations')
    opt = parser.parse_args()
    print(opt)
    check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.detector_weights in ['yolov3.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt']:
                detect()
                strip_optimizer(opt.detector_weights)
        else:
            detect()
