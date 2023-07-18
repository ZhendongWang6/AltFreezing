import cv2
import torch
import numpy as np
import os

def images2video(file, images, fps=30):
    # print(images.shape)
    size = images[0].shape[:2][::-1]
    # print(images.shape)
    # 可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    videoWriter = cv2.VideoWriter(file, fourcc, fps, size)  # 最后一个是保存图片的尺寸
    for i,image in enumerate(images):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        image = image[:,:,::-1]
        image = (image*1.).astype(np.uint8)
        videoWriter.write(image)
        cv2.imwrite(file.replace('.avi', f'_{str(i).zfill(3)}.jpg'), image)
    videoWriter.release()

def save_mask(path, masks):
    for i, mask in enumerate(masks):
        # print(i, np.count_nonzero(mask))
        cv2.imwrite(os.path.join(path, f'{str(i).zfill(3)}_mask.jpg'), (mask * 255.0).astype(np.uint8))
