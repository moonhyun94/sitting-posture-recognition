import wandb
import argparse
import json

import torch
from torch.utils.data import DataLoader

from dataloader.video_loader2 import VideoDataset
from src.train_video2 import *
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
    parser.add_argument('-t', '--fine_tunning_interval', default=1, type=int, help="Fine-tuning interval. default: 1")
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
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        })

    print(json.dumps({
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
    }, indent=4))
        
    dataset = VideoDataset()
    print('dataset loaded')

    if args.model == 'Conv_5_C3D':
        model = Conv_5_C3D(num_classes=6, pretrained=True)
    elif args.model == 'Conv_4_C3D':
        model = Conv_4_C3D(num_classes=6, pretrained=True)
    elif args.model == 'Conv_3_C3D':
        model = Conv_3_C3D(num_classes=6, pretrained=True)
        
    if args.frz:
        print('freezing layers')
        for name, params in model.named_parameters():
            if name not in ['fc6.weight', 'fc6.bias', 'fc7.weight', 'fc7.bias', 'fc8.weight','fc8.bias']:
                params.requires_grad = False
    
    train_dataloader = DataLoader(VideoDataset(split='train',clip_len=30), batch_size=args.batch_size, shuffle=True)
    test_dataloader  = DataLoader(VideoDataset(split='test', clip_len=30), batch_size=args.batch_size)

    train_size = len(train_dataloader.dataset)
    test_size = len(test_dataloader.dataset) # 1234
    
    print('dataloader loaded')
    
    train_video(args, experiment, model, train_dataloader, test_dataloader, train_size, test_size)
