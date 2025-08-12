import sys
import cv2
import numpy as np

def cross_correlation(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    # dimensions of the image and the kernel
    image_height, image_width = img.shape[:2]
    kernel_height, kernel_width = kernel.shape[:2]

    # identifying number of color channels in the image
    channels = 0
    if len(img.shape) == 3:
        channels = img.shape[2]
    elif len(img.shape) == 2:
        channels = 1

    if channels == 1:
        # padding for one channel
        imagePadded = np.zeros((image_height + kernel_height - 1, image_width + kernel_width - 1)) 
        for i in range(image_height):
            for j in range(image_width):
                imagePadded[i + int((kernel_height - 1) / 2), j + int((kernel_width - 1) / 2)] = img[i, j]  

        # applying the kernel
        output = np.zeros_like(img)
        for i in range(image_height):
            for j in range(image_width):
                output[i, j] = np.sum(kernel * imagePadded[i:i + kernel_height, j:j + kernel_width])
    else:
        # padding for multiple channels
        imagePadded = np.zeros((image_height + kernel_height - 1, image_width + kernel_width - 1, channels)) 
        for k in range(channels):
            for i in range(image_height):
                for j in range(image_width):
                    imagePadded[i + int((kernel_height - 1) / 2), j + int((kernel_width - 1) / 2), k] = img[i, j, k]  

        # applying the kernel
        output = np.zeros_like(img)
        for k in range(channels):
            for i in range(image_height):
                for j in range(image_width):
                    output[i, j, k] = np.sum(kernel * imagePadded[i:i + kernel_height, j:j + kernel_width, k])

    return output

def convolution(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    output = cross_correlation(img, np.flip(kernel))

    return output

def gaussian_kernel(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''

    kernel = np.zeros((height, width))
    # center coordinates of the kernel
    center_y = height // 2
    center_x = width // 2

    for i in range(height):
        for j in range(width):
            kernel[i, j] = (1/(2*np.pi*sigma**2)) * np.exp(-((i-center_y)**2 + (j-center_x)**2) / (2*sigma**2))
    
    # normalizing the kernel
    kernel = kernel / np.sum(kernel)

    return kernel

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    kernel = gaussian_kernel (sigma, size, size)

    return cross_correlation (img, kernel)

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    low_pass_img = low_pass (img, sigma, size)

    return img - low_pass_img

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)

def main():
    img1_path = 'C:/Users/User/Downloads/--_left.png'
    img2_path = 'C:/Users/User/Downloads/--_lab02_right.png'
    output_path = 'C:/Users/User/Downloads/hybrid.png'

    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # parameters
    sigma1 = 300.0  
    size1 = 35
    high_low1 = 'low'
    sigma2 = 5.0  
    size2 = 7    
    high_low2 = 'high'
    mixin_ratio = 0.8
    scale_factor = 1.0

    hybrid_img = create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2, high_low2, mixin_ratio, scale_factor)
         
    cv2.imwrite(output_path, hybrid_img) 
    cv2.imshow('Hybrid Image', hybrid_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
