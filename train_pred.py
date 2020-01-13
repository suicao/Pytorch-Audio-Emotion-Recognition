import torch
import numpy as np
import pandas as pd
from model_dcase import ConvNet
from model_m1 import Classifier_M2, Classifier_M3
from model_m0 import Classifier
from sklearn.model_selection import StratifiedKFold
from joblib import load, dump
from sklearn.metrics import accuracy_score
import os
import numpy as np
import cv2
from PIL import Image
import PIL
from torchvision.transforms import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastprogress import master_bar, progress_bar
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
import time
import pickle
import argparse
import random
from tqdm import tqdm
from utils import *
from mixup import *

parser = argparse.ArgumentParser()
parser.add_argument('--train_df_path', type=str, default="../input/train_label.csv")
parser.add_argument('--train_processed_path', type=str, default="../../erc2019/data/mels_train.pkl")
parser.add_argument('--test_processed_path', type=str, default="../../erc2019/data/mels_test.pkl")
parser.add_argument('--test_dir', type=str, default="../input/Public_Test/Public_Test/")
parser.add_argument('--output_dir', type=str, default="./preds")
parser.add_argument('--output_name', type=str, default="preds_tmp.npy")
parser.add_argument('--logdir', type=str, default="./models")
parser.add_argument('--model', type=str, default="m0")

parser.add_argument('--tta', type=int, default=12)
parser.add_argument('--epochs', type=int, default=71)
parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--ckpt_ensemble', type=int, default=0)
parser.add_argument('--mixup', type=int, default=1)
parser.add_argument('--cutmix', type=int, default=0)
parser.add_argument('--predict_only', type=int, default=0)

args = parser.parse_args()

if not os.path.exists(args.logdir):
    os.mkdir(args.logdir)
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

SEED = 420
bs = 64
sz = 128

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(SEED)

model_dict = {
    "m0": Classifier,
    "m2": Classifier_M2,
    "m3": Classifier_M3,
    "dcase": ConvNet,
}
Model = model_dict[args.model]

train_df = pd.read_csv(args.train_df_path)
processed = pickle.load(open(args.train_processed_path, "rb"))
test_fns = sorted(os.listdir(args.test_dir))
test_df = pd.DataFrame()
test_df["File"] = test_fns
processed_test = pickle.load(open(args.test_processed_path, "rb"))

transforms_dict = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ]),
}


x_train = processed
y_train = train_df.Label.values.astype(np.long)
x_test = processed_test
splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(x_train, y_train))


def train_model(x_train, y_train, train_transforms):
    num_epochs = args.epochs
    batch_size = args.batch_size
    lr = 1e-3
    eta_min = 1e-5
    t_max = 10

    num_classes = 6
    train_dataset = ERCTrainDataset(x_train, y_train, train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = Model(num_classes=num_classes).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = Adam(params=model.parameters(), lr=lr, amsgrad=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    mb = master_bar(range(num_epochs))
    for epoch in mb:
        model.train()
        for x_batch, y_batch in progress_bar(train_loader, parent=mb):
            if args.mixup or args.cutmix:
                if args.mixup and (not args.cutmix):
                    x_batch, y_batch_a, y_batch_b, lam = mixup_data(x_batch, y_batch)
                elif args.cutmix and (not args.mixup):
                    x_batch, y_batch_a, y_batch_b, lam = cutmix_data(x_batch, y_batch)
                else:
                    x_batch, y_batch_a, y_batch_b, lam = cutmix_data(x_batch, y_batch) if np.random.rand() > 0.5 else mixup_data(x_batch, y_batch)
                preds = model(x_batch.cuda())
                loss = mixup_criterion(criterion,preds, y_batch_a.cuda(), y_batch_b.cuda(), lam)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                preds = model(x_batch.cuda())
                loss = criterion(preds, y_batch.cuda())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        scheduler.step()
    torch.save(model.state_dict(), f"{args.logdir}/model_full_{args.model}.pt")


if not args.predict_only:
    if args.mixup:
        print("Training using mixup")
    if args.cutmix:
        print("Training using cutmix")
    train_model(x_train, y_train, transforms_dict['train'])


test_batch_size = args.batch_size
num_classes = 6
model = Model(num_classes=num_classes).cuda()
all_preds = []

if args.ckpt_ensemble:
    print("Using checkpoing ensemble for prediction...")
print("Predicting...")
print(f"./{args.logdir}/model_full_{args.model}.pt")
model.load_state_dict(torch.load(f"./{args.logdir}/model_full_{args.model}.pt"))
preds_tta = []
for tta in tqdm(range(args.tta)):
    test_dataset = ERCTrainDataset(x_test, None, transforms_dict['test'], spec_aug=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    test_preds = np.zeros((len(x_test), num_classes))
    for i, x_batch in enumerate(test_loader):
        preds = model(x_batch.cuda()).detach()
        test_preds[i * test_batch_size: (i + 1) * test_batch_size] = preds.cpu().numpy()
    preds_tta.append(test_preds)
all_preds.append(np.mean(preds_tta, axis=0))

np.save(os.path.join(args.output_dir, args.output_name), all_preds[-1])
