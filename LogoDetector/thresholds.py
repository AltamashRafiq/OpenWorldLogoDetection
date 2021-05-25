import argparse
import torch
from pathlib import Path
from utils.torch_utils import select_device
import time
import pickle
import os
import numpy as np


def run_assertions() -> None:
    """Check all input assertions."""
    if opt.T is not None:
        assert (opt.train is None) and (opt.known is None) and (
            opt.unknown is None), "none of --train, --known, and --unknown should be provided if testing preformance with --T"
        assert (opt.test is not None) and (
            opt.test_unknown is not None), "if testing threshold performance, --test and --test-unknown are required arguments"
    else:
        assert opt.name is not None, "name of the output thresholds file (--name) must be provided to save thresholds"
        assert os.path.exists("config/" + opt.name +
                              ".p") == False, f"File config/{opt.name}.p already exists!"
        assert (opt.train is not None) and (opt.known is not None) and (
            opt.unknown is not None), "--train, --known, and --unknown are required arguments"
    if opt.test is not None:
        assert opt.test_unknown is not None, "if testing threshold performance, --test and --test-unknown are required arguments"
    if opt.test_unknown is not None:
        assert opt.test is not None, "if testing threshold performance, --test and --test-unknown are required arguments"


def calculate_thresholds(train_labs: torch.Tensor, train_preds: torch.Tensor, train_distances: torch.Tensor,
                         known_labs: torch.Tensor, known_preds: torch.Tensor, known_distances: torch.Tensor,
                         unknown_preds: torch.Tensor, unknown_distances: torch.Tensor) -> dict:
    """
    Calculate thresholds given opt.train, opt.known, and opt.unknown. Thresholds are
    calculated based on within class accuracy.
    """
    t1 = time.time()
    print("Calculating Thresholds... ", end="")
    preds = torch.cat([train_preds[:, 0], known_preds[:, 0], unknown_preds[:, 0]]
                      ).long()  # nearest lookup table neigbhors
    preds2 = torch.cat([train_preds[:, 0], known_preds[:, 0], unknown_preds[:, 1]]
                       ).long()  # second nearest lookup table neigbhors second
    labs = torch.cat([train_labs, known_labs, torch.ones(
        unknown_preds.shape[0], dtype=int).to(0) * -1]).long()  # true labels
    # distances to nearest lookup table neigbhors
    distances = torch.cat([train_distances[:, 0], known_distances[:, 0], unknown_distances[:, 0]])
    # distances to second nearest lookup table neigbhors
    distances2 = torch.cat([train_distances[:, 0], known_distances[:, 0], unknown_distances[:, 1]])

    Ts = {}  # init thresholds
    for l in range(train_labs.max() + 1):  # loop over all classes (except unknown!)
        y_hat = preds  # y_hat = predictions to use for threshold calculation
        real_distances = distances[labs == l]  # distances of true observations
        # distances of unknown observations
        unknown_distances = distances[(labs == -1) & (y_hat == l)]
        if unknown_distances.shape[0] == 0:
            # if no unknown predictions
            y_hat = preds2  # update y_hat to be second nearest neighbor
            unknown_distances = distances2[(labs == -1) & (y_hat == l)]  # update unknown_distances
        if unknown_distances.shape[0] == 0:
            # if still no unknown predictions
            Ts[l] = (real_distances.max() + 2 * torch.std(real_distances)).item()
        else:
            # if at least 1 unknown prediction
            ds = torch.cat([real_distances, unknown_distances])
            # true labels of observations in class l and true labels of unknown observations incorrectly labeled as class l
            true_labs = torch.cat([labs[labs == l], labs[(labs == -1) & (y_hat == l)]])
            # predicted labels of observations in class l and predicted labels of unknown observations incorrectly labeled as class l
            class_preds = torch.cat([y_hat[labs == l], y_hat[(labs == -1) & (y_hat == l)]])
            best_acc = 0
            best_T = 0
            for T in np.linspace(0, int(real_distances.max() * 5), num=501):
                # choose threshold to maximize accuracy within class
                upd_preds = torch.ones(class_preds.shape[0], dtype=int).to(
                    0) * -1  # init predictions
                passed = ds < T  # if distance within class threshold, label as known
                # passed observations are given predicted label, rest remain unknown
                upd_preds[passed] = class_preds[passed]
                acc = (upd_preds == true_labs).sum() / true_labs.shape[0]  # within class accuraxcy
                if acc > best_acc:
                    # if within class accuracy increases, update threshold
                    best_acc = acc
                    best_T = T
                if best_acc >= 0.9999:
                    # if perfect accuracy (large distance between known and unknown observations), set midpoint threshold
                    sep = (unknown_distances.min() - real_distances.max()).item()
                    best_T = best_T + 0.5 * sep
                    break
            Ts[l] = best_T.item()
    print(f"DONE ({time.time() - t1:.2f}s)")
    return Ts


def save_thresholds(Ts: dict) -> None:
    """Save calculated thresholds as pickle file"""
    with open("config/" + opt.name + ".p", "wb") as f:
        pickle.dump(Ts, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Thresholds saved to config/{opt.name}.p\n")


def read_thresholds() -> dict:
    """Read in input thresholds"""
    with open(opt.T, "rb") as f:
        Ts = pickle.load(f)
    return Ts


def read_files(folder: str, device: str) -> None:
    """
    Read in true labels, predicted labels, and distances from predicted labels from input folder
    """
    folder = str(Path(folder))
    labs = torch.load(folder + '/true_labs.pt', map_location=torch.device(device))
    preds = torch.load(folder + '/pred_labs.pt', map_location=torch.device(device))
    distances = torch.load(folder + '/pred_distances.pt', map_location=torch.device(device))
    return labs, preds, distances


def get_thresholds() -> tuple:
    """
    Returns per class thresholds and threshold calculation or performance calculation files.
    See read_files for types of files returned.
    """
    # check assertions
    run_assertions()
    # Select device
    device = select_device(opt.device)

    files = {}  # folder: (labs, preds, distances)
    for folder in [opt.train, opt.known, opt.unknown, opt.test, opt.test_unknown]:
        if folder is not None:
            files[folder] = read_files(folder, device)

    if opt.T is not None:
        # just read thresholds if --T
        Ts = read_thresholds()
    elif opt.train is not None:
        # calculate and save thresholds from scratch
        train_labs, train_preds, train_distances = files[opt.train]
        known_labs, known_preds, known_distances = files[opt.known]
        _, unknown_preds, unknown_distances = files[opt.unknown]

        # calculate thresholds
        Ts = calculate_thresholds(train_labs, train_preds, train_distances,
                                  known_labs, known_preds, known_distances,
                                  unknown_preds, unknown_distances)
        save_thresholds(Ts)  # save thresholds

    return Ts, files


def report_unknown_performance(upd_preds: torch.Tensor, labs: torch.Tensor) -> None:
    """
    Prints:
      1. Percentage of unknown observations correctly identified as unknown.
      2. Percentage of known observations incorrectly classified as unknown.
      3. Micro Top 1 Accuracy: Prediction accuracy on only known observations.
    """
    correct_unknowns = (100 * (upd_preds[labs == -1] ==
                               labs[labs == -1]).sum() / labs[labs == -1].shape[0]).item()
    incorrect_unknowns = (100 * ((upd_preds == -1) & (labs != -1)
                                 ).sum() / (labs != -1).sum()).item()
    micro_acc = (100 * (labs[labs != -1] == upd_preds[labs != -1]
                        ).sum() / labs[labs != -1].shape[0]).item()
    print(f"Correctly Identified Unknown Percentage (Openset Accuracy): {correct_unknowns:.2f}%")
    print(f"Incorrectly Called Unknown: {incorrect_unknowns:.2f}%")
    print(f"Micro Top 1 Accuracy: {micro_acc:.2f}%\n")


def print_performance(Ts: dict, labs: torch.Tensor, preds: torch.Tensor, distances: torch.Tensor):
    """Calculate and print performance"""
    upd_preds = torch.ones(labs.shape[0], dtype=int).to(0) * -1  # init all observations as unknown
    for l in range(labs.max() + 1):  # loop over all classes (except unknown!)
        passed = distances[preds == l] < Ts[l]  # if distance within class threshold, label as known
        to_upd = preds[(preds == l)].clone()
        to_upd[~passed] = -1
        upd_preds[preds == l] = to_upd
    report_unknown_performance(upd_preds, labs)


def performance(Ts: dict, files: list) -> None:
    """Prints performance of thresholds for train and/or test inputs"""
    if opt.train is not None:
        print("TRAIN PERFORMANCE")
        train_labs, train_preds, train_distances = files[opt.train]
        known_labs, known_preds, known_distances = files[opt.known]
        _, unknown_preds, unknown_distances = files[opt.unknown]

        # combine opt.train, opt.known, opt.unknown
        preds = torch.vstack([train_preds, known_preds, unknown_preds]).long()[:, 0]
        labs = torch.cat([train_labs, known_labs, torch.ones(
            unknown_preds.shape[0], dtype=int).to(0) * -1]).long()
        distances = torch.cat([train_distances, known_distances, unknown_distances])[:, 0]
        print_performance(Ts, labs, preds, distances)

    if opt.test is not None:
        print("TEST PERFORMANCE")
        test_labs, test_preds, test_distances = files[opt.test]
        _, test_unknown_preds, test_unknown_distances = files[opt.test_unknown]

        # combine opt.test, opt.test_unknown
        preds = torch.vstack([test_preds, test_unknown_preds]).long()[:, 0]
        labs = torch.cat([test_labs, torch.ones(
            test_unknown_preds.shape[0], dtype=int).to(0) * -1]).long()
        distances = torch.cat([test_distances, test_unknown_distances])[:, 0]
        print_performance(Ts, labs, preds, distances)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=str, default=None,
                        help="path to pre-calculated thresholds for testing")
    parser.add_argument('--name', type=str, default=None,
                        help="name of thresholds file config/<name>.p")
    parser.add_argument('--train', type=str, default=None,
                        help="folder of training activation vectors and labels")
    parser.add_argument('--known', type=str, default=None,
                        help="folder of held out known activation vectors and labels")
    parser.add_argument('--unknown', type=str, default=None,
                        help="folder  held out unknown activation vectors and labels")
    parser.add_argument('--test', type=str, default=None,
                        help="folder of test activation vectors and labels")
    parser.add_argument('--test-unknown', type=str, default=None,
                        help="folder of test unknown activation vectors and labels")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    print(opt)
    print("")
    Ts, files = get_thresholds()
    performance(Ts, files)
