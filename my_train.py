"""Stage 2 for Chest X-ray disease recognition: fine classification"""

from __future__ import print_function, division

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import models, transforms
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from sklearn.metrics import confusion_matrix
import time
import os
import copy
import json
import pdb
import random
import torch.nn.functional as F


os.environ['CUDA_VISIBLE_DEVICES']='4'


manualSeed = random.randint(1, 1000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(manualSeed)


transform = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class ShoulderDataset(Dataset):
    def __init__(self, split, transform=None):
        self.classes = ('健康', '偏上','偏下','偏左','偏右','肩胛骨')
        self.num_classes = len(self.classes)
        if split == 'train':
            set_type = 'train'
        else:
            set_type = 'dev'
        self.set_type = set_type
        print('Creating {} set...'.format(set_type))
        print('=====================================')

        if self.set_type == 'train':
            self.json_file = 'jsons/demo_train.json'
        else:
            self.json_file = 'jsons/demo_test.json'
        assert osp.exists(self.json_file), 'Path does not exist: {}'.format(self.json_file)
        anno = json.load(open(self.json_file))
        images = []
        labels = []
        cnt = 0
        for entry in anno:
            cnt += 1
            img = cv2.imread(entry['data_path'], 1)
            if entry['label'] == 5:
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
                images.append(cv2.resize(img[:256,:,:], (224, 224), interpolation=cv2.INTER_CUBIC))
            else:
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                images.append(img)
            labels.append(entry['label'])
        print('Total images: {}'.format(cnt))
        self.transform = transform
        self.images = images  # torch.from_numpy(images)
        self.labels = labels  # torch.from_numpy(labels)

    def __getitem__(self, index):
        label = torch.Tensor(self.num_classes).zero_()
        label[self.labels[index]] = 1

        if self.transform is not None:
            img = self.transform(self.images[index])
        return img, label

    def __len__(self):
        # print ('\tcalling Dataset:__len__')
        return len(self.images)


image_datasets = {x: ShoulderDataset(x, transform=transform[x]) for x in ['train', 'val']}
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=1, shuffle=True, num_workers=4),
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=16, shuffle=False, num_workers=4),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}


def eval_model(model):
    model.eval()

    img_label = []
    img_pred = []
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        #half_batch_size = int(len(inputs)/2)
        batch_size = int(len(inputs))

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        labels = np.argmax(labels.cpu().numpy(), 1)     # convert one hot to number
        preds = preds.cpu().numpy()

        for i in range(batch_size):
            img_pred.append(preds[i])
            img_label.append(labels[i])
    img_label = np.array(img_label)
    img_pred = np.array(img_pred)
    acc = np.sum(img_label==img_pred)/len(img_label)
    print(confusion_matrix(img_label, img_pred))
    return acc

'''
my code
'''
def transfer_label(labels,l):
    new_labels = torch.Tensor(l).zero_()
    for i in range(len(new_labels)):
        new_labels[i] = np.argmax(labels[i])

    return new_labels

def create_mask(labels,l):
    masks = abs(torch.zeros([l,6],dtype=torch.float32))
    for i in range(l):
        class_idx = np.argmax(labels[i])
        if class_idx == 0:
            masks[i,:] = 1
        elif class_idx ==1 or class_idx ==2:
            masks[i,0:3] = 1
        elif class_idx ==3 or class_idx==4:
            masks[i,3:5] = 1
            masks[i,0] = 1
        elif class_idx ==5:
            masks[i,0] = 1
            masks[i,-1] =1

    return masks 



def my_loss(inputs,labels,masks):
    temp = torch.sigmoid(inputs)*masks
    zero = torch.masked_select(temp,masks.byte())
    zero_num = torch.numel(zero)
    num = torch.numel(labels)-zero_num    
    criterion = nn.BCELoss(reduction='sum')
    loss = criterion(temp,labels)/num
    
    return loss

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        # for phase in ['train', 'val']:
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                pdb.set_trace()
                inputs = inputs.to(device)
                labels = labels.to(device)
                l = len(labels)
                masks = create_mask(labels,l)
                masks = masks.cuda()
                #labels = transfer_label(labels,l)
                #labels = labels.long().cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    
                    #outputs = torch.sigmoid(outputs)
                    #loss = F.binary_cross_entropy(outputs, labels, weight=masks)
                    #loss = criterion(outputs, labels)
                    loss = my_loss(outputs,labels,masks)
                    #pdb.set_trace()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    #m = nn.Softmax(dim=0)
                    #outputs = m(outputs)
                    #outputs = torch.sigmoid(outputs)
                    preds = outputs.data.max(1,keepdim=True)[1]
                    labels = transfer_label(labels,l)
                    labels = labels.long().cuda()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += preds.eq(labels.data.view_as(preds)).cpu().sum()


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            #epoch_acc = 0

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            torch.save(model.state_dict(), 'training/my_classification_best_model.pkl')
            best_model_wts = copy.deepcopy(model.state_dict())

            '''
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, 'training/my_best_model.pkl')
            '''

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
print(model_ft.fc)
model_ft.fc = nn.Linear(num_ftrs, image_datasets['train'].num_classes)

print(model_ft.fc)
model_ft = model_ft.to(device)

#criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()

#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
optimizer_ft = optim.Adam(model_ft.parameters(),lr=0.001, betas=(0.9,0.999))
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

'''
if os.path.exists('training/my_classification_best_model.pkl'):
    model_ft.load_state_dict(torch.load('training/my_classification_best_model.pkl'))
'''

trained_model = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)

acc = eval_model(trained_model)
print("Acc on dev set: {}".format(acc))
