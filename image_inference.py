import cv2

from FastSAM2.fastsam import FastSAM, FastSAMPrompt
import torch

model = FastSAM('FastSAM-x.pt')
IMAGE_PATH = 'FastSAM2/examples/dogs.jpg'
img = cv2.imread(IMAGE_PATH)
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
everything_results = model(
    IMAGE_PATH,
    device=DEVICE,
    retina_masks=True,
    imgsz=1024,
    conf=0.4,
    iou=0.9,
)
prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

# # everything prompt
ann = prompt_process.everything_prompt()

frame = prompt_process.plot_to_result(annotations=ann)
cv2.imshow('frame', frame)
cv2.waitKey(0)