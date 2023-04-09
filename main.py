import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style='darkgrid')


def Emboss(img):
    # Define kernel for emboss effect
    Emboss_Kernel = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
    # Apply filter2D function to the image with the defined kernel
    Emboss_Effect_Img = cv2.filter2D(src=img, kernel=Emboss_Kernel, ddepth=-1)
    # Create a figure with two subplots, one for the original image and one for the edited image
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    # Plot the original image in the first subplot
    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    # Plot the edited image in the second subplot
    axs[1].imshow(Emboss_Effect_Img, cmap="gray")
    axs[1].set_title("Emboss Effect Image")
    axs[1].axis("off")
    plt.show()


def Sharpen(img):
    # Define kernel for sharpen effect
    Sharpen_Kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # Apply filter2D function to the image with the defined kernel
    Sharpen_effect_Img = cv2.filter2D(src=img, kernel=Sharpen_Kernel, ddepth=-1)
    # Create a figure with two subplots, one for the original image and one for the edited image
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    # Plot the original image in the first subplot
    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    # Plot the edited image in the second subplot
    axs[1].imshow(Sharpen_effect_Img, cmap="gray")
    axs[1].set_title("Sharpen Effect Image")
    axs[1].axis("off")
    plt.show()


def Blur(img):
    # Apply GaussianBlur function to the image with a kernel size of (35, 35) and sigma value of 0
    Blur_Effect_Img = cv2.GaussianBlur(img, (35, 35), 0)
    # Create a figure with two subplots, one for the original image and one for the edited image
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    # Plot the original image in the first subplot
    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    # Plot the edited image in the second subplot
    axs[1].imshow(Blur_Effect_Img, cmap="gray")
    axs[1].set_title("Blur Effect Image")
    axs[1].axis("off")
    plt.show()


def Sepia(img):
    # Define kernel for sharpen effect
    Sepia_Kernel = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
    # Apply the kernel to the image using cv2.transform
    Sepia_effect_Img = cv2.transform(image, Sepia_Kernel)
    # Combine the sepia image with the original color image using weighted addition
    Sepia_effect_Img = cv2.addWeighted(img, 0.5, Sepia_effect_Img, 0.5, 1)
    # Create a figure with two subplots, one for the original image and one for the edited image
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    # Plot the original image in the first subplot
    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    # Plot the edited image in the second subplot
    axs[1].imshow(Sepia_effect_Img, cmap="gray")
    axs[1].set_title("Sepia Effect Image")
    axs[1].axis("off")
    plt.show()


# Load the image
image = cv2.imread("puppies.jpeg")
# Convert the image from BGR to RGB format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Define a function to switch between the three image effects
def switch_effect(effect, img):
    # Define a dictionary with keys as the effect names and values as the corresponding function names
    effects_dict = {"emboss": Emboss, "sharpen": Sharpen, "blur": Blur, "sepia": Sepia}
    # Call the function corresponding to the selected effect
    effects_dict[effect](img)


# Prompt the user to select an effect
print("Please select an effect to apply: ")
print("1. Emboss")
print("2. Sharpen")
print("3. Blur")
print("4. Sepia")
effect = input("Enter the corresponding number: ")

# Call the switch_effect function with the selected effect and the loaded image
switch_effect({
"1": "emboss",
"2": "sharpen",
"3": "blur",
"4": "sepia"
}[effect], image)
