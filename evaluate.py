from scipy import stats
from argparse import ArgumentParser
import torch
from PIL import Image
from pmiqd import model,Crop
import numpy as np
import os
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_dir", type=str, default='')
    parser.add_argument("--model_file", type=str, default='/',)
    args = parser.parse_args()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    model = model().to(device)
    model.load_state_dict(torch.load(args.model_file))
    model.eval()
    scores = []
    img_list = os.listdir(args.img_dir)
    img_list.sort()
    img_nums = len(img_list)
    im_names = img_list
    print(im_names)
    with torch.no_grad():
        for i in range(len(im_names)):
            im = Image.open(os.path.join(args.img_dir,im_names[i])).convert('RGB')
            data = Crop(im)
            dist_patches = data.unsqueeze(0).to(device)
            score = model((dist_patches))
            scores.append(score.item())
            print(score.item())
    scorespath='/'
    np.save(scorespath, scores)

    print(scores)
    # plcc = stats.pearsonr(scores, mos)[0]
