# grabcut_bgremoval.py
import cv2
import numpy as np
from PIL import Image

def remove_background_grabcut(pil_image):
    """
    Removes the background from an image using the GrabCut algorithm.
    Input: PIL Image
    Output: Image (with transparent background)
    """
    # Convert PIL to OpenCV format
    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mask = np.zeros(image.shape[:2], np.uint8)

    # Create background and foreground models
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)

    # Define a rectangle covering almost the entire image
    height, width = image.shape[:2]
    rect = (10, 10, width - 10, height - 10)

    # Apply GrabCut
    cv2.grabCut(image, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
    
    # Convert mask to binary (0 or 1)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Apply mask to image
    result = image * mask2[:, :, np.newaxis]
    
    # Convert to RGBA (make background transparent)
    b, g, r = cv2.split(result)
    alpha = np.where(mask2 == 0, 0, 255).astype('uint8')
    rgba = cv2.merge((r, g, b, alpha))
    
    # Convert back to PIL for Streamlit
    return Image.fromarray(rgba)

