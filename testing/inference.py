import torch 
from model.model import RCNN, validate_batch
import cv2 as cv 
import matplotlib.pyplot as plt 
import sys
sys.path.append('../')
from utils import * 
from main import rcnn, test_ds
from dataset import * 

def test_predications(filename, show_output=True):
    img = np.array(cv.imread(filename, 1)[..., ::-1])
    candidates = extract_candidates(img)
    candidates = [(x,y,x+w,y+h)for x,y,w,h in candidates]
    input = []
    for candidate in candidates : 
        x,y,X,Y = candidate
        crop = cv.resize(img[y:Y, x:X], (224, 224))
        input.append(preprocess_image(crop / 255.)[None])
    input = torch.cat(input).to(device)
    with torch.no_grad():
        rcnn.eval()
        probs, deltas = rcnn(input)
        props = torch.nn.functional.softmax(probs, -1)
        confs,  clss = torch.max(probs, -1)
        candidates = np.array(candidates)
        confs, clss, probs,deltas = [tensor.detach().cpu().numpy() for tensor in [confs, clss, props, deltas]]
        ixs = clss!= background_class

        confs, clss, probs, deltas, candidates = [tensor[ixs] for tensor in [confs, clss, probs, deltas, candidates]]
        bbs = (candidates + deltas).astype(np.uint16)

        ixs = nms(torch.tensor(bbs.astype(np.float32)), torch.tensor(confs), 0.05)
        confs, clss, probs, deltas, candidates, bbs = [
            tensor[ixs] for tensor in [confs, clss, probs, deltas, candidates, bbs]
        ]
        if len(ixs) == 1 :
            confs, clss, probs, deltas, candidates, bbs = [
                tensor[None] for tensor in [confs, clss, probs, deltas, candidates,bbs]
            ]
        # fetch bounding box with height conf 
        if len(confs) == 0 and not show_output:
            return (0,0,224,224), 'background', 0
        if len(confs) > 0 :
            best_pred = np.argmax(confs)
            best_conf = np.max(confs)
            best_bb = bbs[best_pred]
            x,y,X,Y = best_bb
            # plot image with bbox 
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        show(img, ax=ax[0])
        ax[0].set_title('Original Image')
        if len(confs)  == 0 :
                ax[1].imshow(img)
                ax[1].set_title('No Object')
                plt.show()
                return 
         
        ax[1].set_title(target2label[clss[best_pred]])
        show(img, bbs=bbs.tolist(), texts=[target2label[c] for c in clss.tolist()],
             ax = ax[1], title='predicted boundig box and class')
        plt.show()
        return (x,y,X,Y), target2label[clss[best_pred]], best_conf 
    
if __name__ == '__main__':
     image, crops, labels, deltas, gttbs, fpath = test_ds[7]
     test_predications(fpath)