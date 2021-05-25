import argparse
from pathlib import Path
import torch
from torch import nn
import torchvision
import time
from torchvision import datasets
from utils.torch_utils import select_device
from utils.general import increment_path
from utils.classification import load_classifier, classify, classifier_transforms
import os
import pickle as pkl


def get_sz() -> None:
    """Return transformation size for given timm model"""
    if opt.classifier == 'efficientnet_b0':
        return 224
    elif opt.classifier == 'efficientnet_b3':
        return 300


def knn(labs: torch.Tensor, preds: torch.Tensor, distances: torch.Tensor,
        avs: torch.Tensor, outputs: torch.Tensor, targets: torch.Tensor,
        train_avs: torch.Tensor, train_labs: torch.Tensor, K: int, start: int) -> tuple:
    """
    Update input labels and activation vectors with those of preds and outputs. Also update
    input distances with distances of
    """
    # update existing labels and activation vectors
    if labs is not None:
        labs = torch.cat([labs, targets])
        avs = torch.vstack([avs, outputs])
    else:
        labs = targets.clone()
        avs = outputs.clone()
    # update distances with distances of new activation vectors
    for av in outputs:
        dist = torch.norm(train_avs - av, dim=1, p=None)  # distances to all lookup table entries
        nearest = dist.topk(K, largest=False)  # K nearest labels
        if preds is not None:
            # update predictions (nearest K labels)
            preds = torch.vstack([preds, train_labs[nearest.indices[start:]]])
            distances = torch.vstack([distances, nearest.values[start:]])  # update distances
        else:
            # init predictions and distances
            preds = train_labs[nearest.indices[start:]].clone()
            distances = nearest.values[start:].clone()
    return labs, preds, distances, avs


def train(classifier: nn.Sequential, device: str, loader: torch.utils.data.DataLoader) -> tuple:
    """
    Do open set classification on all training set images. Return train
    activation vectors and associated labels.
    """
    print("Calculating train activation vectors... ", end="")
    t1 = time.time()
    avs = None
    labels = None
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = classify(classifier, opt.open_type, inputs)  # open set classification
            if avs is not None:
                # update labels and avs
                labels = torch.cat([labels, targets])
                avs = torch.vstack([avs, outputs])
            else:
                # init labels and avs
                labels = targets.clone()
                avs = outputs.clone()
    print(f"DONE ({time.time() - t1:.2f}s)\n")
    return avs, labels


def test(classifier: nn.Sequential, train_avs: torch.Tensor, train_labs: torch.Tensor,
         device: str, loader: torch.utils.data.DataLoader, n: str) -> tuple:
    """
    Do open set classification on all non-training set images. Return calculated
    activation vectors, associated true labels, K nearest neighbor predictions, and
    associated distances.
    """
    print(
        f"Calculating predicted labels and distances for {n}... ", end="")  # n = name of the data set
    t1 = time.time()
    # inits
    labs = None
    preds = None
    distances = None
    avs = None
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = classify(classifier, opt.open_type, inputs)  # open set classification
            labs, preds, distances, avs = knn(
                labs, preds, distances, avs, outputs, targets, train_avs, train_labs, opt.K, 0)
    print(f"DONE ({time.time() - t1:.2f}s)\n")
    return labs, preds, distances, avs


def save_train_files(train_class_dict: dict, train_avs: torch.Tensor,
                     train_labs: torch.Tensor, save_dir: str) -> None:
    """
    Save train activation vectors, labels, and class dictionary to save_dir.
    """
    direc = str(save_dir) + "/"
    if os.path.exists(direc) == False:
        os.mkdir(direc)
    torch.save(train_avs, direc + 'train_avs.pt')  # activation vectors
    torch.save(train_labs, direc + 'train_labs.pt')  # labels
    with open(direc + 'train_class_dict.p', 'wb') as f:
        # class dictionary
        pkl.dump(train_class_dict, f, protocol=pkl.HIGHEST_PROTOCOL)


def read_train_files(device: str) -> tuple:
    """
    Read train activation vectors, labels, and class dictionary from command line input sources.
    """
    print(
        f"Reading in train activation vectors and labels from '{opt.train_avs}' and '{opt.train_labs}'")
    train_avs = torch.load(opt.train_avs, map_location=torch.device(device))  # activation vectors
    train_labs = torch.load(opt.train_labs, map_location=torch.device(device))  # labels
    with open(opt.train_class_dict, 'rb') as f:
        # class dictionary
        train_class_dict = pkl.load(f)
    return train_class_dict, train_avs, train_labs


def save_files(labs: torch.Tensor, preds: torch.Tensor, distances: torch.Tensor,
               avs: torch.Tensor, save_dir: Path) -> None:
    """
    Save non-train activation vectors, true labels, K nearest distances, and K nearest predictions.
    """
    direc = str(save_dir) + "/"
    if os.path.exists(direc) == False:  # make directory if doesn't exist
        os.mkdir(direc)
    torch.save(labs, direc + 'true_labs.pt')  # ture labels
    torch.save(preds, direc + 'pred_labs.pt')  # K nearest predictions
    torch.save(distances, direc + 'pred_distances.pt')  # K nearest distances
    torch.save(avs, direc + 'avs.pt')  # activation vectors
    print(
        f"True labels, predicted labels (top {opt.K}), associated distances, and activation vectors to {direc}")


def run_assertions() -> None:
    """Check all input assertions."""
    assert opt.classifier in [
        "efficientnet_b0", "efficientnet_b3"], "Options for --classifier are 'efficientnet_b0' or 'efficientnet_b3'"
    assert opt.n_classes > 0, "--n-classes must be specified and be > 0"
    assert opt.open_type in [0, 1, 2], "--open-type must be one of 0, 1, and 2"

    if (opt.train is None):  # if not train source
        # at least one input dataset must exist
        assert (opt.source is not None) or (
            opt.name is not None), "at least one --source must be specified as a valid path and a --name must be provided"
        # open set learning look up table must exist
        assert (opt.train_avs is not None) and (opt.train_labs is not None) and (
            opt.train_class_dict is not None), "Options for --train-avs, --train-labs, and --train-class-dict must be specified if no input for --train"
        # there must as many datasets as there are paths to datasets
        assert len(opt.name) == len(
            opt.source), "the same number of inputs must be provided for opt.name and opt.source."

    if opt.append_classes:
        # train dataset source must be provided and pre-existing lookup table must be linked if appending classes
        assert opt.train is not None, "--train must be provided if --append-classes"
        assert (opt.train_avs is not None) and (opt.train_labs is not None) and (
            opt.train_class_dict is not None), "Options for --train-avs, --train-labs, and --train-class-dict must be specified if --append-classes"


def upd_train_labs(train_labs: torch.Tensor, train_class_dict: dict, new_train_class_dict: dict) -> torch.Tensor:
    """
    If appending classes, train labels in lookup table must be adjusted to match new
    train data set ordering.
    """
    name_dict = {name: ID for ID, name in new_train_class_dict.items()
                 }  # dictionary of brand_name:ID
    new_train_labs = train_labs.clone()  # init new labels
    # update old labels to new
    for i in range(torch.max(train_labs) + 1):
        brand = train_class_dict[i]
        new_train_labs[train_labs == i] = name_dict[brand]
    return new_train_labs


def data_loader(source: str, train_class_dict: dict = None, exclude_classes: dict = None) -> tuple:
    """
    Load in the data to obtain activation vectors. If train_class_dict is provided, the data is
    NOT training data. If exclude_class is provided, the folders with the provided names will be
    skipped when reading in the data.
    """
    print(f"Loading in data from {source}... ", end="")
    transform = classifier_transforms(get_sz(), no_pil=False)  # data transforms
    dataset = datasets.ImageFolder(root=source, transform=transform)  # read in the data
    name_dict = dataset.class_to_idx  # brand_name:ID dictionary
    lab_dict = {ID: name for name, ID in name_dict.items()}  # ID:brand_name dictionary
    if train_class_dict is not None:
        # if NOT training data
        rev_train_class_dict = {name: ID for ID, name in train_class_dict.items()}
        # correct labels for non training data if size mismatch
        dataset.samples = [(s[0], rev_train_class_dict[lab_dict[s[1]]]) for s in dataset.samples]
    if exclude_classes is not None:
        # exclude the brands already completed
        to_keep = list(set(name_dict.keys()) - set(exclude_classes))
        keep_ids = []
        for ID in lab_dict.keys():
            if lab_dict[ID] in to_keep:
                keep_ids.append(ID)
        dataset.samples = [(s[0], s[1]) for s in dataset.samples if s[1] in keep_ids]
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    print("DONE\n")
    return lab_dict, loader


def prepare_open_set() -> None:
    """
    Main function to prepare the open set files for each of the input sources. Can:
    1. Create a train lookup table
    2. Update train lookup table with new n_classes
    3. Return distance metrics to any number of input datasets
    """
    # Check for correct inputs
    run_assertions()
    # Select device
    device = select_device(opt.device)
    # Load classifier
    classifier = load_classifier(opt.classifier, opt.classifier_weights,
                                 device, opt.n_classes, opt.open_type)

    # make directory to save activation vectors: config/activations
    save_dir = Path('config/activations')
    if save_dir.exists() == False:
        os.mkdir('config/activations')
    if opt.project is not None:
        save_dir = increment_path(save_dir / opt.project, exist_ok=False)
        os.mkdir(save_dir)
        save_dir = Path(save_dir)

    # Calculating activation vectors
    train_avs = None
    train_labs = None
    if (opt.append_classes == False) and (opt.train is not None):
        # if making lookup table for the first time
        train_class_dict, loader = data_loader(opt.train)  # load data
        train_class_dict[-1] = 'unknown'  # set -1 label for unknown class
        train_avs, train_labs = train(classifier, device, loader)  # get train avs and labels
        save_train_files(train_class_dict, train_avs, train_labs,
                         save_dir / "train")  # save train avs and labels
        labs, preds, distances, avs = knn(  # get distance metrics of train data with itself
            None, None, None, None, train_avs, train_labs, train_avs, train_labs, opt.K + 1, 1)
        save_files(labs, preds, distances, avs, save_dir / "train")  # save train distance metrics

    if opt.append_classes:
        # if appending classes to train look up table
        print(f"Appending new classes to train activation vectors and labels")
        train_class_dict, train_avs, train_labs = read_train_files(device)  # load older train files
        new_train_class_dict, loader = data_loader(  # load NEW train data
            opt.train, exclude_classes=list(train_class_dict.values()))  # exclude all the classes (folders) previosuly added to lookup table
        train_labs = upd_train_labs(train_labs, train_class_dict,
                                    new_train_class_dict)  # update train labels with new ids
        new_train_avs, new_train_labs = train(
            classifier, device, loader)  # new train activation vectors and labels (to append to lookup table)

        # update lookup table
        train_avs = torch.vstack([train_avs, new_train_avs])
        train_labs = torch.cat([train_labs, new_train_labs])
        train_class_dict = new_train_class_dict  # update train class distionary
        train_class_dict[-1] = 'unknown'
        save_train_files(train_class_dict, train_avs, train_labs,
                         save_dir / "train")  # save new train avs and labels
        labs, preds, distances, avs = knn(  # get distance metrics of train data with itself
            None, None, None, None, train_avs, train_labs, train_avs, train_labs, opt.K + 1, 1)
        save_files(labs, preds, distances, avs, save_dir / "train")  # save train distance metrics

    if opt.source is not None:
        # get activations, labels, predictions, and distances for all non-train datasets
        if train_labs is None:
            # if train files given as input, read them in
            train_class_dict, train_avs, train_labs = read_train_files(device)
        for n, s in zip(opt.name, opt.source):
            # loop over all input data sets and save the results
            _, loader = data_loader(s, train_class_dict)
            labs, preds, distances, avs = test(classifier, train_avs, train_labs, device, loader, n)
            save_files(labs, preds, distances, avs, save_dir / n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', type=str, default=None, help="source of training data")
    parser.add_argument('-s', '--source', action='append', default=None, help='sources')
    parser.add_argument('-n', '--name', action='append', default=None,
                        help='names of directories to store results config/activations/<name> or config/activations/<project>/<name>. See --project')
    parser.add_argument('-a', '--append-classes', action='store_true',
                        help="flag for when new classes are being appended to train activation vectors")
    parser.add_argument('--project', default="project", type=str,
                        help='name of directory to store results config/activations/<project>/<name>')
    parser.add_argument('--train-avs', default=None, type=str,
                        help='path to activation vectors for train data (only needed if not --train)')
    parser.add_argument('--train-labs', default=None, type=str,
                        help='path to labels associated with activation vectors for train data (only needed if not --train)')
    parser.add_argument('--train-class-dict', default=None, type=str,
                        help='path to class dictionary for train data (only needed if not --train)')
    parser.add_argument('--open-type', type=int, default=0,
                        help='1 if last layer of classifier is activation vector, 2 if second last, 0 if concatination of last and second last')
    parser.add_argument('--classifier', type=str, default='efficientnet_b3',
                        help='options: efficientnet_b0 or efficientnet_b3 defaults to efficientnetb0')
    parser.add_argument('--classifier-weights', type=str,
                        default='efficientnet_b3.pt', help='model.pt path(s)')
    parser.add_argument('--n-classes', type=int,
                        default=-1, help='number of classes the classifier is trained on')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--K', type=int,
                        default=2, help='K for K Nearest Neighbors. Must be integer >= 1')
    opt = parser.parse_args()
    print(opt)
    print("")
    prepare_open_set()
