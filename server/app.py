# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect, url_for, abort

# PyTorch関連
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

# Pillow(PIL)、datetime
from PIL import Image, ImageOps
from datetime import datetime

# model
from model import NetworkMNIST as Network
import utils
import argparse

parser = argparse.ArgumentParser("mnist")
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs)')
parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--save', type=str, default='weight', help='experiment name')
parser.add_argument('--model_path', type=str, default='weight/weight.pt', help='path of pretrained model')
args = parser.parse_args()

device = torch.device("cpu")
model = Network().to(device)
utils.load(model,args.model_path)
model = model.eval()

app = Flask(__name__)




@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
         # アプロードされたファイルをいったん保存する
        f = request.files["file"]
        filepath = "./static/" + datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
        f.save(filepath)
        # 画像ファイルを読み込む
        image = Image.open(filepath)
        # PyTorchで扱えるように変換(リサイズ、白黒反転、正規化、次元追加)
        image = ImageOps.invert(image.convert("L")).resize((28, 28))
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        image = transform(image).unsqueeze(0)
        # 予測を実施
        output = model(image)
        _, prediction = torch.max(output, 1)
        result = prediction[0].item()
        
        return render_template("index.html", filepath=filepath, result=result)


if __name__ == '__main__':
    app.run(debug=True)