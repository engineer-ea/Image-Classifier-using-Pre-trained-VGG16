# PROGRAMMER: EZZA ALI

import torch
from torchvision import datasets, transforms
import argparse
from PIL import Image
import numpy as np

def get_train_transforms():
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    
    return train_transforms

def get_test_transforms():
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    return test_transforms

def get_train_data(data_dir):
    train_dir = data_dir + '/train'
    train_transforms = get_train_transforms()
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_data

def get_validation_data(data_dir):
    valid_dir = data_dir + '/valid'
    valid_transforms = get_test_transforms()
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    return valid_data

def get_test_data(data_dir):
    test_dir = data_dir + '/test'
    test_transforms = get_test_transforms()
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data

def get_dataloaders(data_dir):
    batch = 64
    
    train_data = get_train_data(data_dir)
    valid_data = get_validation_data(data_dir)
    test_data = get_test_data(data_dir)
    
    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch)
    
    return trainloader, validloader, testloader


def get_train_args():
    """
    Basic usage: python train.py data_directory
    Options:
        • Set directory to save checkpoints: python train.py data_dir --save_dir
        save_directory
        • Choose architecture: python train.py data_dir --arch "vgg13"
        • Set hyperparameters: python train.py data_dir --learning_rate 0.01 --
        hidden_units 512 --epochs 20
        • Use GPU for training: python train.py data_dir --gpu
    
    Returns an argparse parser.
    """
    archs = [
        'vgg11',
        'vgg13',
        'vgg16',
        'vgg19',
        'densenet121',
        'densenet169',
        'densenet161',
        'densenet201'
    ]
    
    parser = argparse.ArgumentParser(
        description="Train and save an image classification model.",
        usage="python ./train.py ./flowers/train --gpu --learning_rate 0.001 --hidden_units 4096 --epochs 3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('data_directory', action="store", type=str)

    parser.add_argument('--save_dir',
                        action="store",
                        default=".",
                        dest='save_dir',
                        type=str,
                        help='Directory to save training checkpoint file',
                        )

    parser.add_argument('--arch',
                        action="store",
                        default="vgg11",
                        dest='arch',
                        type=str,
                        help='Default: vgg11\nSupported architectures: ' + ", ".join(archs),
                        )

    parser.add_argument('--gpu',
                        action="store_true",
                        dest="gpu",
                        default=False,
                        help='Use GPU? Default: False')

    ag = parser.add_argument_group('hyperparameters')

    ag.add_argument('--learning_rate',
                    action="store",
                    default=0.001,
                    type=float,
                    help='Learning rate - Default:0.001')

    ag.add_argument('--hidden_units', '-hu',
                    action="store",
                    dest="hidden_units",
                    default=[4096, 512],
                    type=int,
                    nargs='+',
                    help='Hidden layer units - Default: 4096, 512')

    ag.add_argument('--epochs',
                    action="store",
                    dest="epochs",
                    default=3,
                    type=int,
                    help='Epochs - Default:3')

    parser.parse_args()
    return parser


def get_predict_args():
    """
    Basic usage: python predict.py /path/to/image checkpoint
    • Options:
        • Return top K most likely classes: python predict.py input checkpoint --
        top_k 3
        • Use a mapping of categories to real names: python predict.py input
        checkpoint --category_names cat_to_name.json
        • Use GPU for inference: python predict.py input checkpoint --gpu
    
    Returns an argparse parser.
    """
    archs = [
        'vgg11',
        'vgg13',
        'vgg16',
        'vgg19',
        'densenet121',
        'densenet169',
        'densenet161',
        'densenet201'
    ]
    
    parser = argparse.ArgumentParser(
        description="Predict an image class using trained model's checkpoint.",
        usage="python predict.py /path/to/image checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('image_path', action="store", type=str)
    parser.add_argument('checkpoint', action="store", type=str)

    parser.add_argument('--category_names',
                        action="store",
                        default="./cat_to_name.json",
                        dest='category_names',
                        type=str,
                        help='Directory to access category names JSON file. Default: ./cat_to_name.json',
                        )

    parser.add_argument('--gpu',
                        action="store_true",
                        dest="gpu",
                        default=False,
                        help='Use GPU? Default: False')

    parser.add_argument('--top_k',
                    action="store",
                    default=5,
                    dest="top_k",
                    type=int,
                    help='Top k predicted class probabilities - Default:5')

    parser.parse_args()
    return parser


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    with Image.open(image) as im:
        
        transform_image = get_test_transforms()
        
        image = transform_image(im)
        return image
    
def plot_results(image_path, top_probs, top_classes, cat_to_name):
    # display image classes and probs
    
    print('\nPlotting prediction results...\n')
    pred_label = top_classes[0]
    pred_image_class = cat_to_name[pred_label]
    
    
    for i in range(len(top_classes)-1):
        print(f'{i+1} - Predicted Class: {cat_to_name[top_classes[i]]}, Probability: {top_probs[i]:.3f}')
        
    print(f'\nPredicted Class for input image: {pred_image_class}')
    