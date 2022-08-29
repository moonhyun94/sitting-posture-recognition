import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.data import CustomDataset
from models.feature_extractor import ConvMixerFeatureExtractor
from models.motion_classifier import MotionClassifier
from sklearn.metrics import accuracy_score


def train_step(context, data):
    motion_classifier = context["motion_classifier"]
    optimizer = context["optimizer"]

    optimizer.zero_grad()

    frames, sensors, labels = data

    result =  motion_classifier(frames, sensors)
    loss = 0.0
    loss = loss + F.cross_entropy(result["logits"], labels)
    loss = loss + F.cross_entropy(result["image_motion_logits"], labels)
    loss = loss + F.cross_entropy(result["sensor_motion_logits"], labels)
    loss.backward()
    optimizer.step()

    fusion_preds = result["logits"].argmax(dim=1).detach().cpu().numpy()
    frame_preds = result["image_motion_logits"].argmax(dim=1).cpu().numpy()
    sensor_preds = result["sensor_motion_logits"].argmax(dim=1).cpu().numpy()

    fusion_acc = accuracy_score(labels.cpu().numpy(), fusion_preds)
    frame_acc = accuracy_score(labels.cpu().numpy(), frame_preds)
    sensor_acc = accuracy_score(labels.cpu().numpy(), sensor_preds)

    return loss, fusion_acc, frame_acc, sensor_acc


@torch.no_grad()
def eval_step(context, data):
    motion_classifier = context["motion_classifier"]

    frames, sensors, labels = data

    result =  motion_classifier(frames, sensors)
    loss = 0.0
    loss = loss + F.cross_entropy(result["logits"], labels)
    loss = loss + F.cross_entropy(result["image_motion_logits"], labels)
    loss = loss + F.cross_entropy(result["sensor_motion_logits"], labels)

    fusion_preds = result["logits"].argmax(dim=1).detach().cpu().numpy()
    frame_preds = result["image_motion_logits"].argmax(dim=1).cpu().numpy()
    sensor_preds = result["sensor_motion_logits"].argmax(dim=1).cpu().numpy()

    fusion_acc = accuracy_score(labels.cpu().numpy(), fusion_preds)
    frame_acc = accuracy_score(labels.cpu().numpy(), frame_preds)
    sensor_acc = accuracy_score(labels.cpu().numpy(), sensor_preds)

    return loss, fusion_acc, frame_acc, sensor_acc


def save(context, path):
    state_dict = dict()

    for key in context.keys():
        if hasattr(context[key], "state_dict"):
            state_dict[key] = context[key].state_dict()

    torch.save(state_dict, path)


def load(context, path):
    state_dict = torch.load(path, map_location="cpu")

    for key in context.keys():
        if hasattr(context[key], "load_state_dict") and key in state_dict.keys():
            context[key].load_state_dict(state_dict[key])


def train(args):
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    trainset = CustomDataset("train", augmentation=True)
    testset = CustomDataset("test")

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=4)

    feature_extractor = ConvMixerFeatureExtractor().to(device).eval()
    # feature_extractor.requires_grad_(False)

    motion_classifier = MotionClassifier(feature_extractor).to(device)
    optimizer = optim.AdamW(motion_classifier.parameters(), lr=args.lr, weight_decay=0.1)

    context = {
        "motion_classifier": motion_classifier,
        "optimizer": optimizer
    }

    # motion_classifier.sensor_features.load_state_dict(torch.load("ckpts/sensor_features.pt"))

    min_test_loss = np.inf

    train_fusion_acc_list = []
    train_frame_acc_list = []
    train_sensor_acc_list = []
    test_fusion_acc_list = []
    test_frame_acc_list = []
    test_sensor_acc_list = []

    for e in range(args.epochs):
        train_loss = 0.0
        train_fusion_acc = 0.0
        train_frame_acc = 0.0
        train_sensor_acc = 0.0

        test_loss = 0.0
        test_fusion_acc = 0.0
        test_frame_acc = 0.0
        test_sensor_acc = 0.0

        motion_classifier.train()

        for frames, sensors, labels in tqdm(train_loader, desc="Training"):
            frames = frames.to(device)
            sensors = sensors.to(device)
            labels = labels.long().to(device)

            loss, fusion_acc, frame_acc, sensor_acc = train_step(context, [frames, sensors, labels])
            train_loss += loss / len(train_loader)
            train_fusion_acc += fusion_acc / len(train_loader)
            train_frame_acc += frame_acc / len(train_loader)
            train_sensor_acc += sensor_acc / len(train_loader)

        motion_classifier.eval()

        for frames, sensors, labels in tqdm(test_loader, desc="Evaluating"):
            frames = frames.to(device)
            sensors = sensors.to(device)
            labels = labels.long().to(device)

            loss, fusion_acc, frame_acc, sensor_acc = eval_step(context, [frames, sensors, labels])
            test_loss += loss / len(test_loader)
            test_fusion_acc += fusion_acc / len(test_loader)
            test_frame_acc += frame_acc / len(test_loader)
            test_sensor_acc += sensor_acc / len(test_loader)

        print(f"Epochs {e + 1}/{args.epochs}")
        print(f"Train loss: {train_loss:.8f}")
        print(f"Test loss: {test_loss:.8f}")
        print(f"Train fusion acc: {train_fusion_acc:.8f}")
        print(f"Train frame acc: {train_frame_acc:.8f}")
        print(f"Train sensor acc: {train_sensor_acc:.8f}")
        print(f"Test fusion acc: {test_fusion_acc:.8f}")
        print(f"Test frame acc: {test_frame_acc:.8f}")
        print(f"Test sensor acc: {test_sensor_acc:.8f}")

        train_fusion_acc_list.append(train_fusion_acc)
        train_frame_acc_list.append(train_frame_acc)
        train_sensor_acc_list.append(train_sensor_acc)
        test_fusion_acc_list.append(test_fusion_acc)
        test_frame_acc_list.append(test_frame_acc)
        test_sensor_acc_list.append(test_sensor_acc)

        if min_test_loss > test_loss:
            min_test_loss = test_loss
            save(context, "ckpts/motion_classifier_20220829_2.pt")

    train_acc = np.stack([
        np.array(train_fusion_acc_list),
        np.array(train_frame_acc_list),
        np.array(train_sensor_acc_list)
    ], axis=1)

    test_acc = np.stack([
        np.array(test_fusion_acc_list),
        np.array(test_frame_acc_list),
        np.array(test_sensor_acc_list)
    ], axis=1)

    with open("train_acc.csv", "w") as f:
        for i in range(args.epochs):
            f.write(f"{train_acc[i, 0]},{train_acc[i, 1]},{train_acc[i, 2]}\n")

    with open("test_acc.csv", "w") as f:
        for i in range(args.epochs):
            f.write(f"{test_acc[i, 0]},{test_acc[i, 1]},{test_acc[i, 2]}\n")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
