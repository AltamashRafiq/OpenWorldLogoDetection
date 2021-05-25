from torch import nn
import torch
import timm
from PIL import Image
from torchvision import transforms


def load_classifier(classifier, classifier_weights, device, n_classes, av=1):
    classifier_path = classifier_weights  # proper path here
    classifier = timm.create_model(classifier, pretrained=False)
    classifier.classifier = torch.nn.Linear(classifier.classifier.in_features, n_classes)
    classifier.to(device)
    classifier.load_state_dict(torch.load(classifier_path)['m'])
    if av == 2:
        classifier = nn.Sequential(*list(classifier.children())[:-1])
        classifier.eval()
    elif av == 0:
        c1 = nn.Sequential(*list(classifier.children())[:-1])
        c2 = nn.Sequential(list(classifier.children())[-1])
        c1.eval()
        c2.eval()
        classifier = c1, c2
    return classifier


@torch.no_grad()
def classify(classifier, open_type, img):
    if open_type != 0:
        return classifier(img)
    else:
        neural_net, multinomial_regression = classifier
        av2 = neural_net(img)
        av1 = multinomial_regression(av2)
        return torch.hstack([av1, av2])


class resize_and_pad:
  def __init__(self, final_size = 300):
    self.final_size = final_size
  def __call__(self, image):
    original_size = image.size  # Obtain image size
    ratio = float(self.final_size)/max(original_size) # Obtain aspect ratio
    new_size = tuple([int(x*ratio) for x in original_size])
    image = image.resize(new_size, Image.ANTIALIAS) # Resize image based on largest dimension 
    new_im = Image.new("RGB", (self.final_size, self.final_size), color = (255,255,255)) # Create a new image and paste the resized on it
    new_im.paste(image, ((self.final_size-new_size[0])//2,
                        (self.final_size-new_size[1])//2))
    return new_im


def classifier_transforms(sz, no_pil = True):
    if no_pil:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            resize_and_pad(final_size=sz),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform = transforms.Compose([
            resize_and_pad(final_size=sz),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    return transform
