import cv2
import numpy as np

all_img_path = ['Shell001.png', 'Shell002.png', 'Shell003.png']

# Reference coin diameter in mm
reference_diameter_mm = 26.76
mm_to_inch_conversion = 0.0393701  # 1 mm = 0.0393701 inches

# Kernel of ones for dilation and erosion
kernel = np.ones((12, 12), np.uint8)  

for img_path in all_img_path:
    # Read the image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Gaussian blur to reduce noise and smoothen the image
    img_gauss = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Apply Otsu's thresholding
    (T, img_thresh) = cv2.threshold(img_gauss, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Canny edge detection to detect edges in the image
    img_canny = cv2.Canny(img_thresh, 50, 150)
    
    # Dilate and erode to fill in the gaps in the edges
    img_dilate = cv2.dilate(img_canny, kernel, iterations=2)  
    img_erode = cv2.erode(img_dilate, kernel, iterations=1) 
    
    # Find contours
    contours, _ = cv2.findContours(img_erode.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (noise) based on area by including only area greater than 1000 pixels
    filtered_contours = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            filtered_contours.append(cnt)
    contours = filtered_contours
    
    # Sorts the contours by area and assumes the largest contour is the shell and the second largest is the coin
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    shell_contour = contours[0]
    coin_contour = contours[1]

    # Calculate the scale based on the coin contour
    coin_area_pixels = cv2.contourArea(coin_contour)
    coin_diameter_pixels = np.sqrt(4 * coin_area_pixels / np.pi)

    # Calculate the scale (pixels per millimeter)
    scale = coin_diameter_pixels / reference_diameter_mm

    # Calculate the shell area in pixels, convert it to mm then square inches
    shell_area_pixels = cv2.contourArea(shell_contour)
    shell_area_mm2 = shell_area_pixels / (scale ** 2)
    shell_area_in2 = shell_area_mm2 * (mm_to_inch_conversion ** 2)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Draw the shell contour on the image
    cv2.drawContours(img, [shell_contour], -1, (0, 0, 255), 2)

    img = cv2.resize(img, (800, 600))
    cv2.imshow(f"Shell in {img_path}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Shell area in {img_path}: {shell_area_in2:.2f} in²")
