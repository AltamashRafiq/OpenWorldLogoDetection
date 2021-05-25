import argparse
import json
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as patches
import cv2
import os
from utils.torch_utils import select_device
from utils.general import increment_path
from pathlib import Path
import time
from scipy.signal import savgol_filter
import pickle
from typing import Callable


def parse_json(file_path: str, coco: bool = False) -> dict:
    """
    Read and parse COCO annotations or prediction json output from detect.py
    """
    with open(file_path) as f:  # read file
        jlist = json.load(f)
    if coco == True:
        jlist = jlist['annotations']  # extract annotations if COCO
    out = {}
    for b in jlist:
        if b['image_id'] in out.keys():  # if image ID already encountered
            if coco == False:
                # store score and label from predictions from detect.py
                out[b['image_id']]['score'].append(b['score'])
                out[b['image_id']]['label'].append(b['label'])
            else:
                # get box coordinates
                b['bbox'][2] = b['bbox'][0] + b['bbox'][2]
                b['bbox'][3] = b['bbox'][1] + b['bbox'][3]
            out[b['image_id']]['bbox'].append(b['bbox'])  # store image box coordinates
        else:  # if image ID not already encountered (initialize image)
            out[b['image_id']] = {}
            if coco == False:
                out[b['image_id']]['score'] = [b['score']]
                out[b['image_id']]['label'] = [b['label']]
            else:
                b['bbox'][2] = b['bbox'][0] + b['bbox'][2]
                b['bbox'][3] = b['bbox'][1] + b['bbox'][3]
            out[b['image_id']]['bbox'] = [b['bbox']]
    if coco == False:
        for ID in out.keys():
            # convert boxes and scores to numpy array
            out[ID]['bbox'] = np.array(out[ID]['bbox'])
            out[ID]['score'] = np.array(out[ID]['score'])
            # convert label column to float64 with 1 for predicted known and -1 for predicted unknown
            temp = np.array(out[ID]['label'])
            new = np.ones(temp.shape)
            new[temp == 'Unknown'] = -1
            out[ID]['label'] = new
    else:
        for ID in out.keys():
            # convert boxes to numpy array
            out[ID]['bbox'] = np.array(out[ID]['bbox'])
    return out


@njit
def box_area(b: np.array) -> np.array:
    """Return area of input box"""
    return (b[2] - b[0]) * (b[3] - b[1])


@njit
def iou_inter(bb1: np.array, bb2: np.array) -> tuple:
    """Returns itersection area, and iou of the two input bounding boxes"""
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:  # if no overlap
        return 0, 0

    # intersection area
    w = (x_right - x_left)
    h = (y_bottom - y_top)
    intersection_area = w * h

    # compute the area of both AABBs
    bb1_area = box_area(bb1)
    bb2_area = box_area(bb2)

    # iou
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return intersection_area, iou


@njit
def confusion_verbose(y: np.array, y_hat: np.array, labels: np.array = None, iou_thres: float = 0.5) -> tuple:
    """
    Returns (between y (true labels) and y_hat (predictions)):
        1. Counts of true positives, false positives, and false negatives.
        2. Box coordinates of true positives, false positives, and false negatives.
        3. Areas of boxes of true positives, false positives, and false negatives.
    """
    # inits
    tp, fp, fn = 0, 0, 0  # counts
    tps, fns, fps = [], [], []  # boxes belonging to category
    tpa, fna, fpa = [], [], []  # areas of boxes belonging to category
    fp_mark = np.array([True] * y_hat.shape[0])  # indicator for false positives
    for i in range(y.shape[0]):
        t = y[i]
        any_t = False
        for j in range(y_hat.shape[0]):
            p = y_hat[j]
            inter, iou = iou_inter(p, t)  # iou of predicted and true box
            # if iou of prediction and true box is over threshold, mark as NOT false positive
            if iou > iou_thres:
                any_t = True
                fp_mark[j] = False  # do not break here as multiple boxes can be tp
        if any_t == True:
            # increment true positives if box found
            tp += 1
            tps.append(t)
            tpa.append(box_area(t))
        else:
            # increment false negatives if box not found
            fn += 1
            fns.append(t)
            fna.append(box_area(t))
    if labels is not None:
        # remove false positive unknowns if opt.remove-unknowns
        false_positives = y_hat[fp_mark & (labels != -1)]
    else:
        false_positives = y_hat[fp_mark]
    for p in false_positives:
        # build false positive collections
        fps.append(p)
        fpa.append(box_area(p))
        fp += 1
    return tp, fp, fn, tps, fps, fns, tpa, fpa, fna


@njit
def confusion(y: np.array, y_hat: np.array, labels: np.array = None, iou_thres: float = 0.5) -> tuple:
    """
    Returns counts of true positives, false positives, and false
    negatives between y and y_hat (predictions)
    """
    tp, fp, fn = 0, 0, 0  # init counts
    fp_mark = np.array([True] * y_hat.shape[0])  # indicator for false positives
    for i in range(y.shape[0]):
        t = y[i]
        any_t = False
        for j in range(y_hat.shape[0]):
            p = y_hat[j]
            inter, iou = iou_inter(p, t)
            # iou of predicted and true box
            # if iou of prediction and true box is over threshold, mark as NOT false positive
            if iou > iou_thres:
                any_t = True
                fp_mark[j] = False  # do not break here as multiple boxes can be tp
        if any_t == True:
            tp += 1  # increment true positives if box found
        else:
            fn += 1  # increment false negatives if box not found
    if labels is not None:
        # remove false positive unknowns if opt.remove-unknowns
        false_positives = y_hat[fp_mark & (labels != -1)]
    else:
        false_positives = y_hat[fp_mark]
    for i in range(false_positives.shape[0]):
        fp += 1  # increment false positives
    return tp, fp, fn


def array_all(d: dict) -> None:
    """Converts lists in boxes and areas keys of input d to np.array """
    d['boxes'] = np.array(d['boxes'])
    d['areas'] = np.array(d['areas'])


def performance(preds: list, test: dict, confs: list, iou_thres: float,
                remove_unknowns: bool, ens_func: Callable = None) -> tuple:
    """
    Reports performance of predictions at provided confidences (one per model).
    Returns dictionaries of true positive, false positive, and false negative
    statistics as a tuple.
    """
    assert len(preds) <= 2, 'You can only ensemble 2 models!'
    # init collections
    t_positives = {"count": 0, "boxes": [], "areas": []}
    f_positives = {"count": 0, "boxes": [], "areas": []}
    f_negatives = {"count": 0, "boxes": [], "areas": []}

    # loop over all true image ids
    for ID in test.keys():
        y = test[ID]['bbox']
        labels = None
        try:
            # if image as at least one prediction
            if len(preds) == 2:  # ensemble
                # subset y_hats by confidence
                y_hat1 = preds[0][ID]['bbox'][preds[0][ID]['score'] >= confs[0]]
                y_hat2 = preds[1][ID]['bbox'][preds[1][ID]['score'] >= confs[1]]
                # init labels
                labels1 = np.array([]).astype('float64')
                labels2 = np.array([]).astype('float64')
                if remove_unknowns:
                    # populate labels if opt.remove-unknowns
                    labels1 = preds[0][ID]['label'][preds[0][ID]['score'] >= confs[0]]
                    labels2 = preds[1][ID]['label'][preds[1][ID]['score'] >= confs[1]]
                # get ensembled y_hat and labels
                y_hat, labels = ens_func(y_hat1, y_hat2, labels1, labels2)
            else:  # not ensemble
                # subset y_hat by confidence
                y_hat = preds[0][ID]['bbox'][preds[0][ID]['score'] >= confs[0]]
                if remove_unknowns:
                    # populate labels if opt.remove-unknowns
                    labels = preds[0][ID]['label'][preds[0][ID]['score'] >= confs[0]]
            # get performance statistics
            tp, fp, fn, tps, fps, fns, tpa, fpa, fna = confusion_verbose(
                y, y_hat, labels, iou_thres)
            t_positives['count'] += tp
            t_positives['boxes'] += tps
            t_positives['areas'] += tpa
            f_positives['count'] += fp
            f_positives['boxes'] += fps
            f_positives['areas'] += fpa
            f_negatives['count'] += fn
            f_negatives['boxes'] += fns
            f_negatives['areas'] += fna
        except KeyError:
            # if image has no predictions, increment false negative collections
            f_negatives['count'] += y.shape[0]
            temp = [fn for fn in y]
            f_negatives['boxes'] += temp
            f_negatives['areas'] += list(map(lambda x: box_area(x), temp))
    array_all(t_positives)
    array_all(f_positives)
    array_all(f_negatives)
    return t_positives, f_positives, f_negatives


def pr(preds: list, test: dict, confs: list, iou_thres: float,
       remove_unknowns: bool, ens_func: Callable = None) -> tuple:
    """
    Reports precision and recall of predictions at provided confidences (one per model).
    """
    assert len(preds) <= 2, 'You can only ensemble 2 models!'
    tpc, fpc, fnc = 0, 0, 0  # init counts
    for ID in test.keys():
        y = test[ID]['bbox']
        labels = None
        try:
            # if image as at least one prediction
            if len(preds) == 2:  # ensemble
                # subset y_hats by confidence
                y_hat1 = preds[0][ID]['bbox'][preds[0][ID]['score'] >= confs[0]]
                y_hat2 = preds[1][ID]['bbox'][preds[1][ID]['score'] >= confs[1]]
                # init labels
                labels1 = np.array([]).astype('float64')
                labels2 = np.array([]).astype('float64')
                if remove_unknowns:
                    # populate labels if opt.remove-unknowns
                    labels1 = preds[0][ID]['label'][preds[0][ID]['score'] >= confs[0]]
                    labels2 = preds[1][ID]['label'][preds[1][ID]['score'] >= confs[1]]
                y_hat, labels = ens_func(y_hat1, y_hat2, labels1, labels2)
            else:  # not ensemble
                # subset y_hat by confidence
                y_hat = preds[0][ID]['bbox'][preds[0][ID]['score'] >= confs[0]]
                if remove_unknowns:
                    # populate labels if opt.remove-unknowns
                    labels = preds[0][ID]['label'][preds[0][ID]['score'] >= confs[0]]
            tp, fp, fn = confusion(y, y_hat, labels, iou_thres)
            tpc += tp
            fpc += fp
            fnc += fn
        except KeyError:
            # if image has no predictions, increment false negative count
            fnc += y.shape[0]
    if tpc + fpc == 0:
        precision = 1.0  # precision is 1 if no predictions made
    else:
        precision = tpc / (tpc + fpc)
    recall = tpc / (tpc + fnc)
    return precision, recall


def pr_curve_data(preds: list, test: dict, iou_thres: float, remove_unknowns: bool = False,
                  step: float = 0.01, ens_func: Callable = None) -> tuple:
    """Get data for precision and recall curve for individual as well as ensembled models"""
    ps, rs = [], []  # store results
    confs = np.arange(0, 1 + step, step)  # 0 - 1 range by step
    if len(preds) == 1:  # not ensemble
        for conf in confs:
            # get p and r values for each confidence threshold
            p, r = pr(preds, test, [conf], iou_thres, remove_unknowns)
            ps.append(p)
            rs.append(r)
        return confs, ps, rs
    else:  # ensemble
        out_confs = []  # store results
        # get p and r values for each combination of confidence thresholds
        for conf1 in confs:
            for conf2 in confs:
                p, r = pr(preds, test, [conf1, conf2], iou_thres, remove_unknowns, ens_func)
                ps.append(p)
                rs.append(r)
                out_confs.append((conf1, conf2))
        return out_confs, ps, rs


@njit
def ensemble(pred1: np.array, pred2: np.array, labels1: np.array, labels2: np.array) -> tuple:
    """
    Ensemble predictions by accept all predictions from pred1
    and any additional predictions from pred2. If either of labels1
    or labels2 for an image is Unknown, the image is classified
    as Unknown.
    """
    if (pred1.shape[0] == 0) and (pred2.shape[0] == 0):
        # no bounding boxes for either model
        return pred1, labels1
    elif pred1.shape[0] == 0:
        # no bounding boxes for first model
        return pred2, labels2
    # at least one bouding boxe for either model
    new_pred = pred1.copy()  # init results
    new_labels = labels1.copy()
    for i in range(pred2.shape[0]):  # check all boxes in pred2 to find new boxes
        box = pred2[i]
        add = True
        for j in range(pred1.shape[0]):
            done = pred1[j]
            _, iou = iou_inter(box, done)
            if iou > 0.4:
                # new box NOT found (do not add)
                add = False
                if labels2[i] == -1:
                    # update label to unknown if new label Unknown
                    new_labels[j] = -1
                break
        if add == True:
            # add new boxe and label
            new_pred = np.vstack((new_pred, box.copy().reshape(1, -1)))
            if labels1.shape[0] > 0:
                new_labels = np.append(new_labels, labels2[i])
    return new_pred, new_labels


def dist(d: dict, ax: plt.axis, label: str, color: str) -> Callable:
    """Plot historgram of performance e.g. true positives"""
    df = pd.DataFrame(np.log(d['areas']), columns=['log(areas)'])
    return sns.histplot(df, x='log(areas)', ax=ax, alpha=0.3, label=label, color=color, bins=20)


def make_line(ax: plt.axis, i: int, count: float) -> None:
    """Make verticle lines and fixed positions on performance histograms"""
    ax[i].vlines(np.log(10000), 0, count + (count / 3), colors=['blue'])
    ax[i].vlines(np.log(100000), 0, count + (count / 3), colors=['red'])
    ax[i].text(np.log(2000), count + (count / 4), "small")
    ax[i].text(np.log(20000), count + (count / 4), "medium")
    ax[i].text(np.log(200000), count + (count / 4), "large")


def compare_errors(results: list, names: list, save_dir: str, log: bool = True) -> None:
    """
    Plot overlapping histograms of models and their collective performance.
    """
    types = ["True Positives", "False Positives", "False Negatives"]  # plot titles
    colors = ['orange', 'blue', 'red', 'green', 'black', 'purple', 'gold']  # plot colors

    # get max values for tp, fp, and fn among the models
    max_vals = [0, 0, 0]
    for i in range(len(results)):
        for j in range(3):
            if max_vals[j] < results[i][j]['count']:
                max_vals[j] = results[i][j]['count']

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    # plot all histograms
    for i in range(3):
        for j in range(len(results)):
            dist(results[j][i], ax[i], names[j], colors[j])
        ax[i].legend(loc="upper right")
        ax[i].set_xlim([6.5, 14.5])
        ax[i].set_title(types[i])

    # make verticle lines for demarcate small, medium, and large logos
    # thresholds are arbitray and can be adjusted
    make_line(ax, 0, max_vals[0] / 8)
    make_line(ax, 1, max_vals[1] / 8)
    make_line(ax, 2, max_vals[2] / 5)

    plt.tight_layout()
    plt.savefig(save_dir + 'histograms.png')  # save histogram
    print(f"Error histograms saved to {save_dir}histograms.png")


def run_assertions() -> None:
    """Check all assertions in command line inputs"""
    assert len(opt.source) >= 1, "At least one --source is required!"
    if opt.ensemble:
        assert len(opt.source) == 2, 'Users can only ensemble 2 models!'


def save_pr_data(confs: list, ps: list, rs: list, model: str, save_dir: str) -> tuple:
    """Print and save precision and recall data for non-ensemble"""
    df = pd.DataFrame(zip(confs, ps, rs))
    df.columns = 'conf', 'p', 'r'
    df['f1'] = (2 * df['p'] * df['r']) / (df['p'] + df['r'])  # calculate f1 score

    # print confidence, precision, and recall at best f1
    conf, p, r, f1 = df.loc[df['f1'] == df['f1'].max()].values[0]
    print(f"Maximum F1 Score: {f1:.3f} --> conf: {conf:.3f}, precision: {p:.3f}, recall: {r:.3f}")

    # print confidence, precision, and recall at opt.recall
    conf, p, r, f1 = df.loc[df['r'] >= opt.recall].tail(1).values[0]
    print(
        f"At Fixed Recall: {opt.recall} --> conf: {conf:.3f}, precision: {p:.3f}, recall: {r:.3f}")

    # save results
    save_pr = save_dir + model + ".csv"
    df.to_csv(save_pr, index=False)
    print(f"PR Data for {model} saved to {save_pr}\n")
    return ps, rs, conf


def save_enseble_pr_data(confs: list, ps: list, rs: list, save_dir: str) -> tuple:
    """Print and save precision and recall data for ensemble"""
    df = pd.DataFrame(confs)
    df.columns = opt.model_names
    df['p'] = ps
    df['r'] = rs
    df = df.sort_values(by=['r', 'p'], ascending=(False)).groupby('r').head(1)
    df['f1'] = (2 * df['p'] * df['r']) / (df['p'] + df['r'])  # f1 score
    df['smooth_p'] = savgol_filter(df['p'].values, 27, 2)  # smooth precision

    # print confidences, precision, and recall at best f1
    conf1, conf2, p, r, f1, smooth_p = df.loc[df['f1'] == df['f1'].max()].values[0]
    print(
        f"Maximum F1 Score: {f1:.3f} --> conf {opt.model_names[0]}: {conf1:.3f}, conf {opt.model_names[1]}: {conf2:.3f}, precision: {p:.3f}, recall: {r:.3f}")

    # tail because recall is in descending order
    r = df.loc[df['r'] >= opt.recall, 'r'].tail(1).values.item()
    temp = df.loc[df['r'] == r]

    # print confidences, precision, and recall at opt.recall
    conf1, conf2, p, r, f1, smooth_p = temp[temp['p'] == temp['p'].max()].values[0]
    print(
        f"At Fixed Recall: {opt.recall} --> conf {opt.model_names[0]}: {conf1:.3f}, conf {opt.model_names[1]}: {conf2:.3f}, precision: {p:.3f}, recall: {r:.3f}")

    # save results
    save_pr = save_dir + "ensemble.csv"
    df.to_csv(save_pr, index=False)
    print(f"PR Data for Ensemble saved to {save_pr}\n")
    return df['smooth_p'].values, df['r'].values, conf1, conf2


def get_save_dir() -> str:
    """Return directory to save results in"""
    save_dir = Path('runs/error_analysis')
    if save_dir.exists() == False:
        # make runs/error_analysis if doesn't exist
        os.mkdir('runs/error_analysis')
    if opt.project is not None:
        # get path of new results directory
        save_dir = increment_path(save_dir / opt.project, exist_ok=False)
        os.mkdir(save_dir)
        save_dir = Path(save_dir)
    save_dir = increment_path(save_dir / opt.name, exist_ok=False) + \
        "/"
    os.mkdir(save_dir)
    return save_dir


def plot_pr(save_dir: str, prs: dict, ensemble_pr: tuple = None) -> None:
    """Plot precision-recall curve"""
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    for m in prs.keys():  # not ensembled
        ps, rs = prs[m]
        plt.plot(rs, ps, label=m)
    if opt.ensemble:  # ensembled
        ps, rs = ensemble_pr
        plt.plot(rs, ps, label="Ensemble")
    plt.legend()
    plt.title(f"Precision Recall Curve/s - IOU {opt.iou}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim(0, 1)
    plt.savefig(save_dir + 'pr.png')
    print(f"PR Curve saved to {save_dir}pr.png")


def error_heatmaps(results: list, names: list, save_dir: str) -> None:
    """Plot heatmaps of errors for all models + ensemble"""
    types = ["True Positives", "False Positives", "False Negatives"]  # titles
    printable_types = ["tp", "fp", "fn"]  # file name subscripts
    img = np.ones((1440, 2560, 3), dtype='int') * 255  # init img
    for i in range(len(results)):
        tp, fp, fn = results[i]
        boxes = tp['boxes'], fp['boxes'], fn['boxes']
        for j in range(3):
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(img)
            for b in boxes[j]:
                # plot box on img for each box in result category
                rect = patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                         linewidth=0, facecolor='r', alpha=opt.alpha)
                ax.add_patch(rect)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.title(names[i] + " " + types[j])
            plt.savefig(save_dir + names[i] + "_" + printable_types[j] + '.png')
    print(f"Heatmaps saved in {save_dir}")


def save_performance(t_positives: dict, f_positives: dict, f_negatives: dict, save_dir: str, name: str) -> None:
    """Save performance result for one model"""
    results = {'true_pos': t_positives, 'false_pos': f_positives, 'false_neg': f_negatives}
    with open(save_dir + name + '_performance.p', 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def error_analysis() -> None:
    """Main function to do detector focused error analysis"""
    # Check for correct inputs
    run_assertions()
    # make directory to save activation vectors
    save_dir = get_save_dir()
    # loading in test json
    test = parse_json(opt.test, coco=True)

    # loading in predictions from sources
    predictions = {}
    for i in range(len(opt.source)):
        m = opt.model_names[i]
        s = opt.source[i]
        pred = parse_json(s)
        predictions[m] = pred

    # init results collections
    prs = {}
    all_confs = {}

    # calculate precision-recall information for each model
    for i in range(len(predictions)):
        model = opt.model_names[i]
        print(f"Calculating PR Data for {model}... ", end="")
        t1 = time.time()
        confs, ps, rs = pr_curve_data([predictions[model]], test,
                                      iou_thres=opt.iou, step=0.001,
                                      remove_unknowns=opt.remove_unknowns)
        print(f'DONE ({time.time() - t1:.3f}s)')
        ps, rs, conf = save_pr_data(confs, ps, rs, model, save_dir)
        prs[model] = ps, rs
        all_confs[model] = conf

    # calculate precision-recall information for ensemble
    ensemble_pr = None
    if opt.ensemble:
        print(f"Calculating PR Data for Ensemble... ", end="")
        t1 = time.time()
        confs, ps, rs = pr_curve_data([predictions[opt.model_names[0]], predictions[opt.model_names[1]]], test, iou_thres=opt.iou,
                                      step=0.005, ens_func=ensemble,  # custon ensembling function can go here
                                      remove_unknowns=opt.remove_unknowns)
        print(f'DONE ({time.time() - t1:.3f}s)')
        ps, rs, conf1, conf2 = save_enseble_pr_data(confs, ps, rs, save_dir)
        ensemble_pr = ps, rs

    # plot precision-recall curve
    plot_pr(save_dir, prs, ensemble_pr=ensemble_pr)

    if opt.save_performance:
        # init results collections
        results = []
        names = []

        # save performance for each model
        for m in prs.keys():
            conf = all_confs[m]
            t_positives, f_positives, f_negatives = performance(
                [predictions[m]], test, [conf], iou_thres=opt.iou, remove_unknowns=opt.remove_unknowns, ens_func=None)
            results.append([t_positives, f_positives, f_negatives])
            save_performance(t_positives, f_positives, f_negatives, save_dir, m)
            names += [m]
        # save performance for ensemble
        if opt.ensemble:
            t_positives, f_positives, f_negatives = performance(
                [predictions[opt.model_names[0]], predictions[opt.model_names[1]]],
                test, [conf1, conf2], iou_thres=opt.iou, remove_unknowns=opt.remove_unknowns, ens_func=ensemble)
            results.append([t_positives, f_positives, f_negatives])
            save_performance(t_positives, f_positives, f_negatives, save_dir, "ensemble")
            names += ["ensemble"]

        print(f"Performance results saved in {save_dir}")

        if opt.plot_hists:
            compare_errors(results, names, save_dir)  # make performance histograms
        if opt.plot_heatmaps:
            error_heatmaps(results, names, save_dir)  # make performance heatmaps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', type=str, default=None,
                        help='source of test. Must be in coco format.')
    parser.add_argument('-s', '--source', action='append', default=None, help='sources')
    parser.add_argument('-m', '--model-names', action='append', default=None,
                        help='names of the models being tested in the order they appear in sources')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='iou for considering a bounding box a true positive')
    parser.add_argument('--iou-ensemble', type=float, default=0.4,
                        help='iou for considering two boxes to be the same when ensembling')
    parser.add_argument('--remove-unknowns', action='store_true',
                        help='whether or not to remove unknown false positives (to make open set adjusted PR curves)')
    parser.add_argument('--recall', type=float, default=0.87, help='Threshold of recall to print')
    parser.add_argument('--ensemble', action='store_true', help='update all models')
    parser.add_argument('--save-performance', action='store_true',
                        help='save detailed performance results at specified --recall value')
    parser.add_argument('--plot-hists', action='store_true',
                        help='plot historgrams of errors')
    parser.add_argument('--plot-heatmaps', action='store_true',
                        help='plot heatmaps of errors')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='alpha value for oppacity of heatmap colors. Defaults to 0.01')
    parser.add_argument('--name', default="run", type=str,
                        help='name of directory to store results runs/error_analysis/<name>')
    parser.add_argument('--project', default=None, type=str,
                        help='name of directory to store results runs/error_analysis/<project>/<name>')
    opt = parser.parse_args()
    print(opt)
    print("")
    error_analysis()
