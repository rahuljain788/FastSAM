from FastSAM.fastsam import FastSAM, FastSAMPrompt
import torch
import numpy as np
import cv2
import time

model = FastSAM('FastSAM-x.pt')
# IMAGE_PATH = 'FastSAM/examples/dogs.jpg'

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(DEVICE)

cap = cv2.VideoCapture(1)

while cap.isOpened():
    rec, frame = cap.read()
    start = time.perf_counter()

