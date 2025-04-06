import matplotlib.pyplot as plt
import cv2

image1_path = "/Users/martinkraus/Library/CloudStorage/OneDrive-ZápadočeskáuniverzitavPlzni/Dokumenty/škola/DP/pictures/TRACKING_VIDEOS/ai_city_gt1.png"
image2_path = "/Users/martinkraus/Library/CloudStorage/OneDrive-ZápadočeskáuniverzitavPlzni/Dokumenty/škola/DP/pictures/TRACKING_VIDEOS/ai_city_pred1.png"

img1_bgr = cv2.imread(image1_path)
img2_bgr = cv2.imread(image2_path)
img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)


fig, axes = plt.subplots(2, 1, figsize=(img1_rgb.shape[1] / 100, img2_rgb.shape[0] * 2 / 100), dpi=100)

axes[0].imshow(img1_rgb)
axes[0].axis("off")

axes[1].imshow(img2_rgb)
axes[1].axis("off")

plt.tight_layout()
plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0)
plt.savefig("/Users/martinkraus/Downloads/vertical_tight.png", bbox_inches='tight', pad_inches=0)
plt.show()
