import torch
from torch import nn
from torch.utils.data import DataLoader
from models.sensor_C3D import C3D
from dataloader.sensor_loader import CustomDataset
from tqdm import tqdm

from src.train_sensor import *
import argparse
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-verbose', default=0, type=int,
                        help="Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. default: 0")
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
                        help="Set wandb project name")
    args = parser.parse_args()
    
    return args

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
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }, indent=4))

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
    
    for epoch in tqdm(range(args.epochs)):
        print(f'Epoch : {epoch+1} \n--------------------------------')

        train(args, experiment, train_dataloader, my_model, loss_fn, optimizer, device)
        test(args, experiment, test_dataloader, my_model, loss_fn, device)
        
    print('Done')