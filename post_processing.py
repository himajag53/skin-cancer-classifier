import cv2

from matplotlib import pyplot as plt

def gaussian_blur(image, kernel_size=(5, 5)):
    """
    Applies Gaussian blur to the input image.

    Args:
        image (numpy.ndarray): Input image to be blurred.
        kernel_size (tuple): Size of the Gaussian kernel.

    Returns:
        numpy.ndarray: The blurred image.
    """
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred_image

def histogram_equalization(image):
    """
    Applies histogram equalization to the input image to enhance contrast.

    Args:
        image (numpy.ndarray): Input image to be processed.

    Returns:
        numpy.ndarray: Image with enhanced contrast.
    """
     # grayscale image
    if len(image.shape) == 2:
        equalized_image = cv2.equalizeHist(image)

    # color image
    elif len(image.shape) == 3:  
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        equalized_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    else:
        raise ValueError("Image format unsupported!")
    
    return equalized_image

def edge_detection(image, low_threshold=50, high_threshold=150):
    """
    Applies Canny edge detection to the input image.

    Args:
        image (numpy.ndarray): Input image to be processed.
        low_threshold (int): Lower bound for thresholding.
        high_threshold (int): Upper bound for thresholding.

    Returns:
        numpy.ndarray: Image with edges detected.
    """
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges

def display_image(image, title="Image"):
    """
    Displays an image using matplotlib.

    Args:
        image (numpy.ndarray): Image to be displayed.
        title (str): Title of the image plot, default is "Image".
    """
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()
