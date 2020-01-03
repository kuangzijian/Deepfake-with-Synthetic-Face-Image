import matplotlib.pyplot as plt
from generate_synthetic_face import generate_synthetic_face


print("======================= 1. Generate synthetic face using TL-GAN ========================")
image = generate_synthetic_face()
plt.imshow(image)
plt.show()
print("================= A synthetic face was generated using Human face GAN ==================")

print("============= 2. Face swapping on destination image using the synthetic face ===========")
