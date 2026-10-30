import cv2
import matplotlib.pyplot as plt

img = cv2.imread("albino.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


#Flip horizontal (kiri-kanan) dan vertikal (atas-bawah)
flip_h = cv2.flip(img, 1)
flip_v = cv2.flip(img, 0)


plt.figure(figsize=(15,5))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Citra Asli")
plt.axis("off")


plt.subplot(1, 3, 2)
plt.imshow(flip_h)
plt.title("Horizontal Flip")
plt.axis("off")



plt.subplot(1, 3, 3)
plt.imshow(flip_v)
plt.title("Vertical Flip")
plt.axis("off")

plt.tight_layout()
plt.show()