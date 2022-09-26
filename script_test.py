import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import time
import cv2
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import ipywidgets
from IPython.display import display
from jetcam.utils import bgr8_to_jpeg


def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


global start_time
def execute(image):
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    draw_objects(image, counts, objects, peaks)
    image = cv2.resize(image,(640,640))
    image = cv2.flip(image, 1)
    image = cv2.putText(image, "FPS: " + str(int(1.0 / (time.time() - start_time))), (500, 50), cv2.FONT_HERSHEY_SIMPLEX ,  1, (0, 0, 0), 2, cv2.LINE_AA)
    image_w.value = bgr8_to_jpeg(image)


with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

WIDTH = 224
HEIGHT = 224
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))


t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()

print(50.0 / (t1 - t0))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

cap = cv2.VideoCapture(0)

image_w = ipywidgets.Image(format='jpeg')
display(image_w)

while(cap.isOpened()):
    start_time = time.time() # start time of the loop
    ret,frame = cap.read()
    frame = cv2.resize(frame,(224,224))
    frame = execute(frame)
    if ret == True:
        cv2.imshow("Frame",frame)

        if cv2.waitKey(25) & 0xFF == ord(q):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
