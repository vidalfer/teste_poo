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
    #image_w.value = bgr8_to_jpeg(image)


with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose) #carregando a topologia para que se possa obter os keypoints da pose

WIDTH = 224 #largura
HEIGHT = 224 #altura
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda() #sera gerado um tensor com (1,3,224,224) de dimensao

OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth' #instanciando o caminho do modelo ja otimizado
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL)) #carregando o modelo otimizado


mean = torch.Tensor([0.485, 0.456, 0.406]).cuda() #valores de media que serao utilizados para o preprocessamento da imagem
std = torch.Tensor([0.229, 0.224, 0.225]).cuda() #valores de desvio padrao que serao utilizados para o preprocessamento da imagem
device = torch.device('cuda')

#objetos relacionados com o desenho do esqueleto
parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

cap = cv2.VideoCapture("exemploNetCaindo.mp4") #pode esta sendo utilizado um video para teste, mas basta mudar para 0 caso queira usar a camera

#count = 0
while True:
    start_time = time.time() # start time of the loop
    ret,frame = cap.read()
    if ret == True:
        frame = cv2.resize(frame,(224,224))
        execute(frame)
        #count+=1
        #if count <=10: #condicao para testar se esta tudo ok com a estimacao de pose
            #cv2.imwrite(f"framesteste/frame{count}.jpg",frame)
    else:
        print("Captura finalizada")
        break
cap.release()

    
            


