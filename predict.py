# PROGRAMMER: EZZA ALI

import os
import json
from utility_funcs import get_predict_args, plot_results
from model import predict

def main():
    parser = get_predict_args()
    user_args = parser.parse_args()
    # python predict.py ./flowers/test/10/image_07117.jpg ./vgg16_checkpoint.pth --gpu --top_k 5
    
    print(user_args)
    
     # check for image and checkpoint valid path
    if not os.path.isfile(user_args.image_path):
        print(f'Error! Given Image: {user_args.image_path} not found.')
        exit(1)
    elif not os.path.isfile(user_args.checkpoint):
        print(f'Error! Given checkpoint: {user_args.checkpoint} not found.')
        exit(1)
    
    top_probs, top_classes = predict(user_args.image_path, user_args.checkpoint, user_args.gpu, user_args.category_names, user_args.top_k)
    
    with open(user_args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    plot_results(user_args.image_path, top_probs, top_classes, cat_to_name)
    
if __name__ == '__main__':
    main()