import skimage as ski
import matplotlib.pyplot as plt
import os

blunt_folder_path='resized\\SIZE_120_rescaled_max_area_1024\\120_blunt'
gun_folder_path='resized\\SIZE_120_rescaled_max_area_1024\\120_gun'

# image=ski.data.camera()
# plt.imshow(image, cmap='gray')
# plt.show()

# blunt_bloodstain=ski.io.imread("resized\\SIZE_120_rescaled_max_area_1024\\120_blunt\\C1.jpg")
# plt.imshow(bloodstain, cmap='gray')
# plt.show()

for value in [blunt_folder_path, gun_folder_path]:
    for filename in os.listdir(value):
        if filename.endswith(".jpg"):
            image = ski.io.imread(os.path.join(value, filename))
            plt.imshow(image, cmap='gray')
            plt.title(filename)
            plt.show()
        else:
            continue