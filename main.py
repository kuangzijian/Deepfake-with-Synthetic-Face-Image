import matplotlib.pyplot as plt
import os
import cv2
import argparse

import numpy as np

from face_detection import face_detection
from face_points_detection import face_points_detection
from face_swap import warp_image_2d, warp_image_3d, mask_from_points, apply_mask, correct_colours, transformation_from_points
from generate_synthetic_face import generate_synthetic_face


print("======================= 1. Generate synthetic face using TL-GAN ========================")
image = generate_synthetic_face()
plt.imshow(image)
plt.show()
print("================= A synthetic face was generated using Human face GAN ==================")

print("============= 2. Face swapping on destination image using the synthetic face ===========")
