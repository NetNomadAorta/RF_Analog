import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
import torch
from torchvision import datasets, models
from torch.utils.data import DataLoader
import copy
import math
import re
import cv2
import albumentations as A  # our data augmentation library
# remove warnings (optional)
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm # progress bar
from pycocotools.coco import COCO
# Now, we will define our transforms
from albumentations.pytorch import ToTensorV2


# User parameters
SAVE_NAME      = "./Models-OD/RF_Analog-0.model"
USE_CHECKPOINT = True
IMAGE_SIZE     = int(re.findall(r'\d+', SAVE_NAME)[-1] ) # Row and column size 
DATASET_PATH   = "./Training_Data/" + SAVE_NAME.split("./Models-OD/",1)[1].split("-",1)[0] +"/"
NUMBER_EPOCH   = 10000
BATCH_SIZE     = 2 # Default: Work_PC: 2
LEARNING_RATE  = 0.001*BATCH_SIZE # Default: Work_PC: 0.001*BATCH_SIZE

# Transformation Parameters:
BLUR_PROB           = 0.05  # Default: 0.05 
DOWNSCALE_PROB      = 0.25  # Default: 0.10 
NOISE_PROB          = 0.05  # Default: 0.05 
MOTION_BLUR_PROB    = 0.05  # Default: 0.05
ROTATION            = 5     # Default: 5
BRIGHTNESS_CHANGE   = 0.10  # Default: 0.10
CONTRAST_CHANGE     = 0.05  # Default: 0.05
SATURATION_CHANGE   = 0.05  # Default: 0.05
HUE_CHANGE          = 0.05  # Default: 0.05
HORIZ_FLIP_CHANCE   = 0.20  # Default: 0.10
VERT_FLIP_CHANCE    = 0.20  # Default: 0.10



def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}h:{1}m:{2}s".format(int(hours), int(mins), round(sec) ) )


def get_transforms(train=False):
    if train:
        transform = A.Compose([
            # A.Resize(IMAGE_SIZE, IMAGE_SIZE), # I don't include anymore because OD models doesn't discriminate against size
            # A.Rotate(limit=[90,90], always_apply=True),
            A.GaussianBlur(blur_limit = (3,5), p = BLUR_PROB),
            A.Downscale(scale_min = 0.40, scale_max = 0.99, p = DOWNSCALE_PROB),
            A.GaussNoise(var_limit = (1.0, 10.0), p = NOISE_PROB),
            A.MotionBlur(5, p = MOTION_BLUR_PROB),
            A.ColorJitter(brightness = BRIGHTNESS_CHANGE, 
                          contrast = CONTRAST_CHANGE, 
                          saturation = SATURATION_CHANGE, 
                          hue = HUE_CHANGE, 
                          p = 0.1),
            A.HorizontalFlip(p = HORIZ_FLIP_CHANCE),
            A.VerticalFlip(p = VERT_FLIP_CHANCE),
            A.RandomRotate90(p = 0.2),
            A.Rotate(limit = [-ROTATION, ROTATION]),
            ToTensorV2()
        ], bbox_params = A.BboxParams(format = 'coco') )
    else:
        transform = A.Compose([
            # A.Resize(IMAGE_SIZE, IMAGE_SIZE), # our input size can be 600px
            # A.Rotate(limit=[90,90], always_apply=True),
            ToTensorV2()
        ], bbox_params = A.BboxParams(format = 'coco') )
    return transform


class Object_Detection(datasets.VisionDataset):
    def __init__(self, root, split = 'train', transform = None, 
                 target_transform = None, transforms = None):
        # the 3 transform parameters are reuqired for datasets.VisionDataset
        super().__init__(root, transforms, transform, target_transform)
        self.split = split #train, valid, test
        self.coco = COCO(os.path.join(root, split, "_annotations.coco.json")) # annotatiosn stored here
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = [id for id in self.ids if (len(self._load_target(id)) > 0)]
    
    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]['file_name']
        image = cv2.imread(os.path.join(self.root, self.split, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))
    
    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        target = copy.deepcopy(self._load_target(id))
        
        # required annotation format for albumentations
        boxes = [t['bbox'] + [t['category_id']] for t in target]
        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes)
        
        image = transformed['image']
        boxes = transformed['bboxes']
        
        new_boxes = [] # convert from xywh to xyxy
        for box in boxes:
            xmin = box[0]
            xmax = xmin + box[2]
            ymin = box[1]
            ymax = ymin + box[3]
            new_boxes.append([xmin, ymin, xmax, ymax])
        
        boxes = torch.tensor(new_boxes, dtype=torch.float32)
        
        targ = {} # here is our transformed target
        if len(boxes) == 0:
            boxes = torch.zeros((0,4), dtype=torch.float32)
        targ['boxes'] = boxes
        targ['labels'] = torch.tensor([t['category_id'] for t in target], dtype=torch.int64)
        targ['image_id'] = torch.tensor([t['image_id'] for t in target])
        targ['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # we have a different area
        targ['iscrowd'] = torch.tensor([t['iscrowd'] for t in target], dtype=torch.int64)
        return image.div(255), targ # scale images
    def __len__(self):
        return len(self.ids)



# Starting stopwatch to see how long process takes
start_time = time.time()

torch.cuda.empty_cache()


dataset_path = DATASET_PATH



#load classes
coco = COCO(os.path.join(dataset_path, "train", "_annotations.coco.json"))
categories = coco.cats
n_classes = len(categories.keys())
categories

classes = [i[1]['name'] for i in categories.items()]
classes


train_dataset = Object_Detection(root=dataset_path, transforms=get_transforms(True))


# # Lets view a sample
# sample = train_dataset[2]
# img_int = torch.tensor(sample[0] * 255, dtype=torch.uint8)
# plt.imshow(draw_bounding_boxes(
#     img_int, sample[1]['boxes'], [classes[i] for i in sample[1]['labels']], width=4
# ).permute(1, 2, 0))


# lets load the faster rcnn model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_detections_per_img=500)
# model = models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True) # HOW TO MAKE THIS ONE EXIST
in_features = model.roi_heads.box_predictor.cls_score.in_features # we need to change the head
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)


# Runs last save file / checkpoint model
if os.path.isfile(SAVE_NAME):
    checkpoint = torch.load(SAVE_NAME)
if USE_CHECKPOINT and os.path.isfile(SAVE_NAME):
    model.load_state_dict(checkpoint)

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)


images, targets = next(iter(train_loader))
images = list(image for image in images)
targets = [{k:v for k, v in t.items()} for t in targets]
output = model(images, targets) # just make sure this runs without error



device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use GPU to train



model = model.to(device)




# Now, and optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, nesterov=True, weight_decay=1e-4)
# optimizer = torch.optim.AdamW(params, lr = 1e-4) # CHOOSE THIS ONE (AdamW) OR ABOVE (SGD)
# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16, 22], gamma=0.1) # lr scheduler




import sys

def train_one_epoch(model, optimizer, loader, device, epoch):
    model.to(device)
    model.train()
    
    all_losses = []
    all_losses_dict = []
    
    for images, targets in tqdm(loader):
        images = list(image.to(device) for image in images)
        targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets) # the model computes the loss automatically if we pass in targets
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()
        
        all_losses.append(loss_value)
        all_losses_dict.append(loss_dict_append)
        
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping trainig") # train if loss becomes infinity
            print(loss_dict)
            sys.exit(1)
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        
    all_losses_dict = pd.DataFrame(all_losses_dict) # for printing
    print("Epoch {}, l: {:.3f}, l_class: {:.3f}, l_box: {:.3f}, l_rpn_box: {:.3f}, l_obj: {:.4f}".format(
        epoch, 
        np.mean(all_losses),
        all_losses_dict['loss_classifier'].mean(),
        all_losses_dict['loss_box_reg'].mean(),
        all_losses_dict['loss_rpn_box_reg'].mean(),
        all_losses_dict['loss_objectness'].mean()
    ))
    
    return ( np.mean(all_losses), all_losses_dict['loss_objectness'].mean() )


# lr: {:.6f} = optimizer.param_groups[0]['lr'] , but I took it off from above



num_epochs = NUMBER_EPOCH
prev_saved_all_losses = 100
prev_saved_obj_loss = 100
prev_saved_weighted_loss = 100

for epoch in range(num_epochs):
    all_losses, obj_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
    
    weighted_obj_loss = max(-76.23*(obj_loss/all_losses)+9.0656, 2) * obj_loss
    weighted_loss = all_losses + weighted_obj_loss
    
    # Saves model - version 2 - can comment out if wanted
    if weighted_loss < prev_saved_weighted_loss:
        torch.save(model.state_dict(), SAVE_NAME)
        print("   Saved model!\n")
        prev_saved_weighted_loss = weighted_loss
        prev_saved_obj_loss = obj_loss # Not needed, but just curious
        prev_saved_all_losses = all_losses # Not needed, but just curious
    
    # # Saves model
    # if (obj_loss < prev_saved_obj_loss 
    #     or all_losses < (prev_saved_all_losses*0.9) ): # DEfault 0.85
    #     torch.save(model.state_dict(), SAVE_NAME)
    #     print("   Saved model!")
    #     prev_saved_obj_loss = obj_loss
    #     prev_saved_all_losses = all_losses


torch.cuda.empty_cache()

# # Saves model
# torch.save(model.state_dict(), SAVE_NAME)


print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)