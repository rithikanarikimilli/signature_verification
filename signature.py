import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def show_images(img1, img2):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title("Image 1")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.title("Image 2")
    plt.axis('off')

    plt.show()

def match(path1, path2):
    # read the images
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    # Check if images are loaded successfully
    if img1 is None:
        print(f"Error: Unable to load image at {path1}")
        return None
    if img2 is None:
        print(f"Error: Unable to load image at {path2}")
        return None

    # turn images to grayscale
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # resize images for comparison
    img1 = cv2.resize(img1, (300, 300))
    img2 = cv2.resize(img2, (300, 300))

    # Display both images using matplotlib
    show_images(img1, img2)

    # Calculate similarity value
    similarity_value = ssim(img1, img2) * 100  # SSIM value is between 0 and 1, convert to percentage
    similarity_value = "{:.2f}".format(similarity_value)
    
    return float(similarity_value)

# Example usage
ans = match(r"D:\Code\Git stuff\Signature-Matching\assets\1.png",
            r"D:\Code\Git stuff\Signature-Matching\assets\3.png")

if ans is not None:
    print("Similarity Percentage:", ans)
