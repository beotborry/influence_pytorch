#!/bin/zsh

python main.py --method influence --constraint eo --dataset adult --epoch 20 --iteration 20 --scaler 20
python main.py --method influence --constraint eo --dataset adult --epoch 20 --iteration 20 --scaler 25
python main.py --method influence --constraint eo --dataset adult --epoch 20 --iteration 20 --scaler 30