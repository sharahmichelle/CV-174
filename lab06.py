import cv2
import numpy as np
import matplotlib.pyplot as plt

def stitch_images(images, ratio=0.75, reprojection_threshold=4.0):
    
    # Convert all images to the same color space
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 and img.shape[2] == 3 else img for img in images]
    
    # Initialize feature detector (SIFT) - detect keypoints and compute descriptors
    sift = cv2.SIFT_create()
    result = images[0] # Start with the first image
    
    for i in range(1, len(images)):
        # Convert to grayscale
        gray1 = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY) if len(result.shape) == 3 else result
        gray2 = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY) if len(images[i].shape) == 3 else images[i]
        
        # Detect and compute features
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
        
        # FLANN based matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(descriptors1, descriptors2, k=2) 

        # Lowe's ratio test
        good_matches = [m for m, n in matches if m.distance < ratio * n.distance]
       
        # Coordinates of the matched keypoints in the first image.
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        # Coordinates of the matched keypoints in the second image.
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
       
        # Find homography with RANSAC
        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, reprojection_threshold)
            
        # Warp the new image
        h1, w1 = result.shape[:2]
        h2, w2 = images[i].shape[:2]
        
        # Get the corners of both images
        corners_src = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32).reshape(-1, 1, 2)
        corners_dst = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32).reshape(-1, 1, 2)
        corners_dst_transformed = cv2.perspectiveTransform(corners_dst, H)
        
        # Calculate output dimensions
        all_corners = np.concatenate((corners_src, corners_dst_transformed), axis=0)
        min_x, min_y = np.int32(np.floor(all_corners.min(axis=0).ravel() - 0.5))
        max_x, max_y = np.int32(np.ceil(all_corners.max(axis=0).ravel() + 0.5))

        # Adjust homography for translation
        # translation matrix 
        translation = np.array([
            [1, 0, -min_x],
            [0, 1, -min_y],
            [0, 0, 1]
        ])
       
        H_adjusted = translation @ H
        output_width = max_x - min_x
        output_height = max_y - min_y
        
        # Warp the new image
        warped_image = cv2.warpPerspective(images[i], H_adjusted, (output_width, output_height), 
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        # Create output canvas
        panorama = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # These offsets shift the image into positive space so that it fits within the canvas.
        x_offset = -min_x
        y_offset = -min_y
        
        # Calculate the region where we'll place the current result
        # These variables define the region on the canvas (panorama) where the current stitched result (result) will be placed.
        y_start = max(y_offset, 0)
        x_start = max(x_offset, 0)
        y_end = min(y_offset + h1, output_height)
        x_end = min(x_offset + w1, output_width)
        
        # Calculate the corresponding region in the source image
        src_y_start = max(-y_offset, 0)
        src_x_start = max(-x_offset, 0)
        src_y_end = min(output_height - y_offset, h1)
        src_x_end = min(output_width - x_offset, w1)
        
        # Place the current result in the panorama
        panorama[y_start:y_end, x_start:x_end] = result[src_y_start:src_y_end, src_x_start:src_x_end]
        
        # Simple blending
        non_black = np.any(warped_image > 0, axis=2)
        panorama[non_black] = warped_image[non_black]
        
        # Updates the result variable to hold the current state of the panorama.
        result = panorama
    
    # Crop the black borders
    gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY) if len(result.shape) == 3 else result
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Find all non-black points
    non_zero = cv2.findNonZero(thresh)
    # calculate bounding box
    if non_zero is not None:
        x, y, w, h = cv2.boundingRect(non_zero)
        # Adds a small margin around the bounding box to avoid cutting too close to the content.
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(w + 2 * margin, result.shape[1] - x)
        h = min(h + 2 * margin, result.shape[0] - y)
        # Crops the stitched image (result) to the bounding box with the added margin.
        result = result[y:y+h, x:x+w]
    
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR) if len(result.shape) == 3 else result

if __name__ == "__main__":
    image_paths = [
        'IMG_20250304_164212.jpg',
        'IMG_20250304_164218.jpg',
        'IMG_20250304_164226.jpg',
        'IMG_20250304_164234.jpg'
    ]
    
    # Read images
    images = [cv2.imread(path) for path in image_paths]
    images = [img for img in images if img is not None]
    
    # Stitch all images sequentially
    panorama = stitch_images(images)
    
    # Save and display result
    if panorama is not None:
        cv2.imwrite('--_lab06_stitch.png', panorama)
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
        plt.title("Stitched Image")
        plt.axis('off')
        plt.show()
