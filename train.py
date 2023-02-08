# PROGRAMMER: EZZA ALI

import os
import json
from model import create_model, train_save_model
from utility_funcs import get_train_args

def main():
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
    
    parser = get_train_args()
    user_args = parser.parse_args()
    
    print(user_args)
    #Namespace(arch='vgg11', data_directory='./flowers', epochs=3, hidden_units=[4096, 512], learning_rate=0.001, save_dir='.', use_gpu=True)
    #python train.py ./flowers/ --arch "vgg16" --gpu --learning_rate 0.001 --hidden_units 2048 512 --epochs 5
    
     # check for training data directory
    if not os.path.isdir(user_args.data_directory):
        print(f'Error! Data directory {user_args.data_directory} not found.')
        exit(1)
    
    # check for correct arch
    arch = ''
    if user_args.arch.lower() in archs:
        arch = user_args.arch
    else:
        print(f'Error! Given arch: {user_args.arch} is unknown. Using default arch.')    
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    classes_out = len(cat_to_name)
    
    # Train, Test and Save Model
    train_save_model(arch, user_args.data_directory, user_args.learning_rate, user_args.epochs, user_args.gpu, user_args.save_dir, user_args.hidden_units, classes_out)
    
if __name__ == '__main__':
    main()