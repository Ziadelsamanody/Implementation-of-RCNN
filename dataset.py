import selectivesearch
import torch
from torchvision import transforms, models , datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import nms
import pandas as pd 
from torch_snippets import * 
from utils import * 
import numpy as  np 
import cv2 as cv
device = 'cuda' if torch.cuda.is_available() else 'cpu'

IMAGE_ROOT  = 'images/images'
DF_RAW = pd.read_csv('df.csv')
print(DF_RAW.head())


class OpenImages(Dataset):
    def __init__(self, df, image_folder=IMAGE_ROOT):
        self.root = image_folder
        self.df = df
        self.unique_images = df['ImageID'].unique()

    def __len__(self) :
        return len(self.unique_images)
    def __getitem__(self, idx):
        image_id = self.unique_images[idx]
        image_path = f"{self.root}/{image_id}.jpg"
        # convert BGR to RGB
        image = cv.imread(image_path, 1)[..., ::-1]
        h,w ,_ = image.shape
        df = self.df.copy()
        df = df[df['ImageID'] == image_id]
        boxes = df['XMin,YMin,XMax,YMax'.split(',')].values
        boxes = (boxes*np.array([w,h,w,h])).astype(np.uint16).tolist()
        classes = df['LabelName'].values.tolist()
        return image, boxes, classes, image_path
    

ds = OpenImages(df=DF_RAW)
# img, bbx, clss, _ = ds[9]
# show(img , bbs=bbx, texts=clss, sz=10)





FPATHS, GTBBS,CLSS, DELTAS, ROIS, IOUS =[], [], [], [], [], []

N = 500

for idx , (img, bbs, labels, fpath) in enumerate(ds):
    if idx == N :
        break
    # extract candidates from each image 
    H,W ,_ = img.shape
    candidates = extract_candidates(img)
    # convert coord from xy wh to xy x+w y+h
    candidates = np.array([(x,y,x+w,y+h) for x,y,w,h in candidates])
    ious, rois, clss, deltas = [],[],[],[]
    # store iou of all canidantes  with respect to all ground truth for image where bbs is the gt of bbox of diffrent object 
    ious = np.array([[extract_iou(candidate, __bb__) for candidate in candidates] for __bb__ in bbs]).T
    # loop through candidates and store x1 y1 x2 y2 max, min
    for ix, candidate in enumerate(candidates):
        cx,cy, cX,cY = candidate
        # extract iou correspoindig to candinate with respct to gt
        candidate_ious = ious[ix]
        best_iou_at = np.argmax(candidate_ious)
        best_iou = candidate_ious[best_iou_at]
        best_bb = _x,_y,_X, _Y = bbs[best_iou_at]
        if best_iou > 0.3 : 
            clss.append(labels[best_iou_at])
        else : 
            clss.append('background')
        # fetch offset delta transform the current proposal into the canidate that refio prposal 
        # gt - best bb 
        delta = np.array([_x-cx, _y-cy, _X-cX, _Y-cY]) / np.array([W,H,W,H])
        deltas.append(delta)
        rois.append(candidate / np.array([W,H,W,H]))
        # Append paths iou , roi, clss, delta , gt
        FPATHS.append(fpath)
        IOUS.append(ious)
        ROIS.append(rois)
        CLSS.append(clss)
        DELTAS.append(deltas)
        GTBBS.append(bbs)

# Fetch image path names and store all information obtained in a list of listis
FPATHS = [f"{IMAGE_ROOT}/{stem(f)}.jpg" for f in FPATHS]

FPATHS, GTBBS, CLSS, DELTAS , ROIS  = [item for item in [FPATHS, GTBBS, CLSS, DELTAS, ROIS]]

# convert classes to numric by there inices 0 background 1 bus 2 truck
targets = pd.DataFrame(flatten(CLSS), columns=['label'])
label2target = {l:t for t, l in enumerate(targets['label'].unique())}
target2label = {t:l for l,t in label2target.items()}
background_class = label2target['background']
# print(background_class)
# creating training data

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224, 0.225])

