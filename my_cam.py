# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception
import os
import torch
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import pdb
from torch import topk
import torch.nn as nn
import pdb
import json

os.environ['CUDA_VISIBLE_DEVICES']='0'

classes = {0:'正常', 1:'肩胛骨'}

model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))
model.load_state_dict(torch.load('training/shoulder_best_model.pkl'))
print('Model Loaded!')
finalconv_name = 'layer4'
model.cuda().eval()
#pdb.set_trace()

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        pdb.set_trace()
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def run_cam(impath, output_dir):
    im = impath.split('/')[-1]
    
    img_pil = cv2.imread(impath,1)
    img_pil = cv2.resize(img_pil, (512, 512), interpolation=cv2.INTER_CUBIC)
    img_pil = cv2.resize(img_pil[:256,:,:], (224, 224), interpolation=cv2.INTER_CUBIC)
    img_pil = transform(img_pil)
    img_tensor = img_pil.unsqueeze(0)
    img_variable =  img_tensor.cuda()

    final_layer = model.layer4
    activated_features = SaveFeatures(final_layer)
    
    # get the softmax weight
    params = list(model.fc.parameters())
    weight_softmax = np.squeeze(params[0].cpu().data.numpy())
    prediction = model(img_variable)
    pred_probabilities = F.softmax(prediction, dim=1).data.squeeze()
    activated_features.remove()

    cls_prob = float(topk(pred_probabilities, 1)[0].float())
    cls_idx = int(topk(pred_probabilities, 1)[1].int())
    
    if cls_idx == 0: # normal
        prob_pos = 1 - cls_prob
        prob_neg = cls_prob
    else:
        prob_pos = cls_prob
        prob_neg = 1 - prob_pos
    
    print('{:.3f} -> {}'.format(cls_prob, classes[cls_idx]))

    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(activated_features.features, weight_softmax, [0, 1])
    
    # render the CAM and output
    img = cv2.imread(impath,1)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
    img = img[:256,:,:]
    height, width = 256, 512
    
    if cls_idx == 1:
        heatmap_pos = cv2.applyColorMap(cv2.resize(CAMs[1],(width, height)), cv2.COLORMAP_JET)
        result_pos = heatmap_pos * 0.2 + img * 0.6
        cv2.imwrite(os.path.join(output_dir, im[:-4]+'_肩胛骨_'+str(round(prob_pos,3))+'.jpg'), result_pos)
    elif cls_idx == 0:
        heatmap_neg = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
        result_neg = heatmap_neg * 0.2 + img * 0.6
        cv2.imwrite(os.path.join(output_dir, im[:-4]+'_正常_'+str(round(prob_neg,3))+'.jpg'), result_neg)
    
'''
im_path = 'shoulder.jpg'
run_cam(im_path, './')

im_path = 'normal.jpg'
run_cam(im_path, './')
'''

output_dir = '/home/lianjie/zhikong/shoulder_cam_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

json_file = 'jsons/shoulder_test.json'
anno = json.load(open(json_file))
count = 0
for entry in anno:
    img_path = entry['data_path']
    run_cam(img_path,output_dir)
    count += 1
    if count% 10 == 0:
        print(count)
