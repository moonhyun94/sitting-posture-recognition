<<<<<<< HEAD
import wandb
import argparse
import json
from tqdm import tqdm
import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet34_Weights, ResNet18_Weights, VGG16_Weights
from torch.utils.data import DataLoader

from dataloader.sensor_loader2 import CustomDataset
from src.train_sensor import *
=======
import torch
from torch import nn
from torch.utils.data import DataLoader
from models.sensor_C3D import C3D
from dataloader.sensor_loader import CustomDataset
from tqdm import tqdm

from src.train_sensor import *
import argparse
import json
>>>>>>> 60bab893a28460f24cc22e968cc37b850cbcdef8

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-verbose', default=0, type=int,
                        help="Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. default: 0")
<<<<<<< HEAD
    parser.add_argument('-model', default='resnet18', help='Choose model type resnet18, resnet34, vgg16')
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
=======
    parser.add_argument('-model', default='C3D', help='Choose model type Conv_5_C3D, Conv_3_C3D')
    parser.add_argument('-epochs', default=20, type=int, help="Number of epoch to train. default: 5")
    parser.add_argument('-batch_size', default=1, type=int, help="Number of samples per gradient update. default: 1")
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Set learning rate')
    parser.add_argument('-chkt_filename', default='./c3d-pretrained.pth', help="Model Checkpoint filename to save.")
    # parser.add_argument('-t', '--fine_tunning_interval', default=1, type=int, help="Fine-tuning interval. default: 1")
    parser.add_argument('-gn','--gpu_number', default=0, type=int,
                        help='Number of GPU. default: 0')
    # parser.add_argument('-vdp', '--video_data_path', default='./Large_Captcha_Dataset',
    #                     help="Location of video dataset. default: \'./Large_Captcha_Dataset\'")
    parser.add_argument('-sdp', '--sensor_data_path', default='./data/Train',
                        help="Location of sensor dataset. default: \'./Large_Captcha_Dataset\'")
    parser.add_argument('-wandb', default=False, action="store_true",
                        help="Do you wanna use wandb? just give it True! default:False")
    parser.add_argument('-pn', '--project_name', default='sensor',
>>>>>>> 60bab893a28460f24cc22e968cc37b850cbcdef8
                        help="Set wandb project name")
    args = parser.parse_args()
    
    return args

<<<<<<< HEAD

if __name__ == '__main__':
    args = get_args()
    experiment=None

    if args.wandb:
        experiment = wandb.init(
            project=args.project_name, entity='captcha-active-learning-jinro', config={
                'model' : args.model,
                'learning_rate' : args.learning_rate,
                'epochs' : args.epochs,
                'batch_size' : args.batch_size,})

    print(json.dumps({
        "model" : args.model,
=======
if __name__=="__main__":

    args = get_args()
    experiment = None

    if args.wandb:
        experiment = wandb.init(
            project = args.project_name, entity='captcha-active-learning-jinro', config={
                'learning_rate' : args.learning_rate,
                'epochs' : args.epochs,
                'batch_size' : args.batch_size,
            })

    print(json.dumps({
>>>>>>> 60bab893a28460f24cc22e968cc37b850cbcdef8
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }, indent=4))
<<<<<<< HEAD
        

    fcl = nn.Sequential(
            nn.Linear(512, 6),)
    
    # fcl = nn.Sequential(
    #         nn.Linear(512, 256),
    #         nn.ReLU(),
    #         nn.Dropout(p=0.5),
    #         nn.Linear(256,128),
    #         nn.ReLU(),
    #         nn.Dropout(p=0.5),
    #         nn.Linear(128, 6))
    if torch.cuda.is_available():
        device = f'cuda:{args.gpu_number}'
    else:
        device = 'cpu'
    print(f'using device {device}')
    
    if args.model == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = fcl
    elif args.model == 'resnet34':
        model = models.resnet34(pretrained=True)
        model.fc = fcl
    elif args.model == 'vgg16':
        model = models.vgg16(pretrained=True)
        avgpool = nn.AdaptiveAvgPool2d(output_size = (1,1))
        model.avgpool = avgpool
        model.classifier = fcl
    
    model.to(device)
    print('model loaded')
    
    train_dataset = CustomDataset(mode='Train')
    test_dataset = CustomDataset(mode='Test')
    print('dataset loaded')
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader  = DataLoader(test_dataset, batch_size=args.batch_size)
    print('dataloader loaded')
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    loss_fn.to(device)
=======

    if torch.cuda.is_available():
        device = 'cuda'
    print(device)

    my_model = C3D(num_classes=6, pretrained=True).to(device)

    train_dataset = CustomDataset(mode='Train')
    test_dataset = CustomDataset(mode='Test')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(my_model.parameters(), lr=args.learning_rate)
>>>>>>> 60bab893a28460f24cc22e968cc37b850cbcdef8
    
    for epoch in tqdm(range(args.epochs)):
        print(f'Epoch : {epoch+1} \n--------------------------------')

<<<<<<< HEAD
        train_sensor(args, experiment, model, train_dataloader, loss_fn, optimizer, device)
        test_sensor(args, experiment, model, test_dataloader, loss_fn, optimizer, device)

=======
        train(args, experiment, train_dataloader, my_model, loss_fn, optimizer, device)
        test(args, experiment, test_dataloader, my_model, loss_fn, device)
        
>>>>>>> 60bab893a28460f24cc22e968cc37b850cbcdef8
    print('Done')