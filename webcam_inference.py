import cv2
import time
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

cap = cv2.VideoCapture(0)
count =0
while cap.isOpened():
    count+=1
    if count%20 ==0:
        suc, frame = cap.read()
        start = time.perf_counter()

        everything_results = model(
            frame,
            device=DEVICE,
            retina_masks=True,
            imgsz=1024,
            conf=0.4,
            iou=0.9,
        )

        # print(everything_results[0].masks.shape)
        # print(everything_results[0].boxes.shape)
        # print(everything_results[0].boxes[0].xyxy.cpu().numpy())

        # for box in everything_results[0].boxes:
        #     box = box.xyxy.cpu().numpy()[0]
        #     print(box)
        #     cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        prompt_process = FastSAMPrompt(frame, everything_results, device=DEVICE)

        # # everything prompt
        ann = prompt_process.everything_prompt()

        end = time.perf_counter()
        total_time = end - start
        fps = 1 / total_time

        frame = prompt_process.plot_to_result(annotations=ann)
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
cap.release()