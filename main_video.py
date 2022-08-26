import wandb
import argparse
import json

import torch
from torch.utils.data import DataLoader
from torch import nn

from dataloader.video_loader import VideoDataset
from src.train_video import *
from models.video_models import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-verbose', default=0, type=int,
                        help="Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. default: 0")
    parser.add_argument('-model', default='Conv_5_C3D', help='Choose model type Conv_5_C3D, Conv_3_C3D')
    parser.add_argument('-frz', default=False, help='Choose model type Conv_5_C3D, Conv_3_C3D')
    parser.add_argument('-epochs', default=5, type=int, help="Number of epoch to train. default: 5")
    parser.add_argument('-batch_size', default=1, type=int, help="Number of samples per gradient update. default: 1")
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Set learning rate')
    parser.add_argument('-chkt_filename', default='./weights', help="Model Checkpoint filename to save.")
    parser.add_argument('-gn','--gpu_number', default=0, type=int,
                        help='Number of GPU. default: 0')
    parser.add_argument('-vdp', '--video_data_path', default='./Large_Captcha_Dataset',
                        help="Location of video dataset. default: \'./Large_Captcha_Dataset\'")
    parser.add_argument('-sdp', '--sensor_data_path', default='./Large_Captcha_Dataset',
                        help="Location of sensor dataset. default: \'./Large_Captcha_Dataset\'")
    parser.add_argument('-wandb', default=False, action="store_true",
                        help="Do you wanna use wandb? just give it True! default:False")
    parser.add_argument('-pn', '--project_name', required=True,
                        help="Set wandb project name")
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = get_args()
    experiment = None
    if args.wandb:
        experiment = wandb.init(project=args.project_name, entity="captcha-active-learning-jinro", config={
            "model" : args.model,
            "freeze" : args.frz,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gpu" : args.gpu_number,
        })

    print(json.dumps({
            "model" : args.model,
            "freeze" : args.frz,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gpu" : args.gpu_number,
    }, indent=4))
        
    if args.model == 'Conv_5_C3D':
        model = Conv_5_C3D(num_classes=6, pretrained=True)
    elif args.model == 'Conv_4_C3D':
        model = Conv_4_C3D(num_classes=6, pretrained=True)
    elif args.model == 'Conv_3_C3D':
        model = Conv_3_C3D(num_classes=6, pretrained=True)

    if args.frz == True:
        print('freezing layers')
        for name, params in model.named_parameters():
            if name not in ['fc6.weight', 'fc6.bias', 'fc7.weight', 'fc7.bias', 'fc8.weight','fc8.bias']:
                params.requires_grad = False
    
    if torch.cuda.is_available():
        device = f'cuda:{args.gpu_number}'
    else:
        device = 'cpu'
        
    print('using device', device)
    
    criterion = nn.CrossEntropyLoss()
    
    model.to(device)
    criterion.to(device)
    
    train_dataset = VideoDataset(split='train', clip_len=30)
    test_dataset = VideoDataset(split='test', clip_len=30)
    print('dataset loaded')
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader  = DataLoader(test_dataset, batch_size=args.batch_size)
    print('dataloader loaded')
    
    train_video(args, experiment, model, train_dataloader, test_dataloader, criterion, device)
