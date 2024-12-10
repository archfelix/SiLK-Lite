import os
import torch
import cv2 as cv
from model import SiLK
import utils

device = None
if torch.cuda.is_available():
    print("CUDA is available! Training on GPU.")
    device = torch.device("cuda:0")
else:
    print("CUDA is not available. Training on CPU.")
    device = torch.device("cpu")

model = SiLK()
model = model.to(device)
model.train(True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
save_every_n_image = 1000

image_root_dir = "X:/COCO/train2017/train2017"

if __name__ == "__main__":
    img_files = os.listdir(image_root_dir)
    # index = torch.randperm(len(img_files))
    index = torch.arange(len(img_files))
    count = 0
    for i in index:
        file = img_files[i]
        img = cv.imread(os.path.join(image_root_dir, file), cv.IMREAD_GRAYSCALE)
        # img = cv.resize(img, (320, 240))
        # img = cv.resize(img, (160, 120))
        img = cv.resize(img, (80, 60))
        # img = cv.resize(img, (40, 30))
        cv.imshow("img", img)
        img_tensor = torch.from_numpy(img).to(torch.float32).to(device)
        img_tensor = torch.unsqueeze(img_tensor, dim=0)
        img_tensor = torch.unsqueeze(img_tensor, dim=0)

        optimizer.zero_grad()
        loss_total, loss_desc_num, loss_kpts_num, kpts_count = utils.compute_loss(model, img_tensor, block_size=1200, tau=0.1)
        loss_total.backward()
        optimizer.step()

        print(f"LossDesc={loss_desc_num :.8f}   LossKpts={loss_kpts_num :.8f}   Kpts={kpts_count}")

        cv.waitKey(1)

        count += 1
        if count % save_every_n_image == 0:
            torch.save(model.state_dict(), f"./train0_{count}.pth")
            count = 0
