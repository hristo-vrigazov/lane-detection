import math
import numpy as np
from imutils import auto_canny

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# by a given line mx + b
# returns tuple (x1, y1, x2, y2)
# where f(x1) = y1, f(x2) = y2
def compute_line(m, b, x1, x2):
    return int(x1), int(m*x1 + b), int(x2), int(m*x2 + b)

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    if lines is None or len(lines) == 0:
        return img
    
    left_line_m_sum = 0
    left_line_b_sum = 0
    number_of_left_lines = 0
    
    right_line_m_sum = 0
    right_line_b_sum = 0
    number_of_right_lines = 0
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                # skip if the line is perpendicular to screen
                continue
            m = ((y2 - y1) / (x2 - x1))
            b = y1 - m * x1
            if m > 0:
                left_line_m_sum += m
                left_line_b_sum += b
                number_of_left_lines += 1
            else:
                right_line_m_sum += m
                right_line_b_sum += b
                number_of_right_lines += 1
    
    averaged_lines = []
    width = img.shape[1]
    
    if number_of_left_lines > 0:
        left_m = left_line_m_sum / float(number_of_left_lines)
        left_b = left_line_b_sum / float(number_of_left_lines)

        left_line = [compute_line(left_m, left_b, width, width*0.55)]
        averaged_lines.append(left_line)

    if number_of_right_lines > 0:
        right_m = right_line_m_sum / float(number_of_right_lines)
        right_b = right_line_b_sum / float(number_of_right_lines)

        right_line = [compute_line(right_m, right_b, -width*0.55, width*0.45)]
        averaged_lines.append(right_line)

    
    for averaged_line in averaged_lines:
        for x1,y1,x2,y2 in averaged_line:
            cv2.line(img, (x1, y1), (x2, y2), color, 15)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def create_region_of_interest(img_shape):
    width = img_shape[1]
    height = img_shape[0]
    
    down_left = (0, height)
    middle = (int(width/2), int(height/2))
    down_right = (width, height)
    
    polygon = np.array([down_left, middle, down_right])
    return np.array([polygon])

def lane_detection_pipeline(img, gaussian_kernel_size=5, rho=1, theta=np.pi/60.0, threshold=70, 
                            min_line_len=40, max_line_gap=50):
    result = grayscale(img)
    result = gaussian_blur(img, gaussian_kernel_size)
    result = auto_canny(result)
    result = region_of_interest(result, create_region_of_interest(img.shape))
    lines_image = hough_lines(result, rho, theta, threshold, min_line_len, max_line_gap)
    return weighted_img(lines_image, img)