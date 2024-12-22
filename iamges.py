import cv2
import numpy as np

def enhance_image(image):
    """Apply various image enhancement techniques"""
    # Convert to float32 for better processing
    img_float = image.astype(np.float32) / 255.0
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    
    # Enhance edges using unsharp masking
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    unsharp_mask = cv2.addWeighted(enhanced, 2.0, gaussian, -1.0, 0)
    
    return unsharp_mask

def detect_edges_multiple(image):
    """Apply multiple edge detection methods"""
    edges_list = []
    
    # Canny edge detection with different parameters
    edges1 = cv2.Canny(image, 30, 150)
    edges2 = cv2.Canny(image, 50, 200)
    
    # Sobel edge detection
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edges3 = np.sqrt(sobelx*2 + sobely*2).astype(np.uint8)
    
    # Laplacian edge detection
    edges4 = cv2.Laplacian(image, cv2.CV_64F).astype(np.uint8)
    
    # Combine all edges
    edges_list.extend([edges1, edges2, edges3, edges4])
    combined_edges = np.maximum.reduce(edges_list)
    
    return combined_edges

def detect_boxes(image):
    try:
        original_image = image.copy()
        results = []
        
        # 1. Image Enhancement
        enhanced = enhance_image(image)
        
        # 2. Multi-scale processing
        scales = [0.5, 1.0, 1.5]
        all_boxes = []
        
        for scale in scales:
            # Resize image
            if scale != 1.0:
                scaled_dim = (int(image.shape[1] * scale), int(image.shape[0] * scale))
                scaled_image = cv2.resize(enhanced, scaled_dim)
            else:
                scaled_image = enhanced
            
            # Apply multiple edge detection methods
            edges = detect_edges_multiple(scaled_image)
            
            # Apply different thresholding methods
            _, binary1 = cv2.threshold(scaled_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            adaptive1 = cv2.adaptiveThreshold(scaled_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            adaptive2 = cv2.adaptiveThreshold(scaled_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Combine binary images
            binary_combined = cv2.bitwise_or(cv2.bitwise_or(binary1, adaptive1), adaptive2)
            
            # Enhance edges
            kernel = np.ones((3,3), np.uint8)
            binary_combined = cv2.dilate(binary_combined, kernel, iterations=1)
            binary_combined = cv2.erode(binary_combined, kernel, iterations=1)
            
            # Find contours using different methods
            contours_modes = [cv2.RETR_EXTERNAL, cv2.RETR_LIST, cv2.RETR_TREE]
            contours_methods = [cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_TC89_KCOS]
            
            all_contours = []
            for mode in contours_modes:
                for method in contours_methods:
                    contours, _ = cv2.findContours(binary_combined, mode, method)
                    all_contours.extend(contours)
            
            # Process each contour
            for contour in all_contours:
                # Scale contour back to original size
                if scale != 1.0:
                    contour = (contour / scale).astype(np.int32)
                
                area = cv2.contourArea(contour)
                if area < 100:  # Adjust threshold as needed
                    continue
                
                # Try different approximation methods
                peri = cv2.arcLength(contour, True)
                approx1 = cv2.approxPolyDP(contour, 0.02 * peri, True)
                approx2 = cv2.approxPolyDP(contour, 0.04 * peri, True)
                
                # Check if either approximation gives us a reasonable shape
                for approx in [approx1, approx2]:
                    if len(approx) >= 4 and len(approx) <= 8:
                        rect = cv2.minAreaRect(approx)
                        box = cv2.boxPoints(rect)
                        box = np.int32(box)  # Updated to np.int32 instead of np.int0
                        
                        # Calculate box properties
                        width = rect[1][0]
                        height = rect[1][1]
                        aspect_ratio = min(width, height) / max(width, height)
                        
                        if aspect_ratio > 0.1:
                            all_boxes.append(box)
        
        # Remove duplicate boxes
        final_boxes = []
        for box in all_boxes:
            is_duplicate = False
            for existing_box in final_boxes:
                # Compare center points
                box_center = np.mean(box, axis=0)
                existing_center = np.mean(existing_box, axis=0)
                if np.linalg.norm(box_center - existing_center) < 20:  # Adjust threshold
                    is_duplicate = True
                    break
            if not is_duplicate:
                final_boxes.append(box)
        
        # Draw and store results
        for i, box in enumerate(final_boxes):
            # Calculate box properties
            rect = cv2.minAreaRect(box)
            x, y = rect[0]
            angle = rect[2]
            
            # Draw box with random color for better visualization
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            cv2.drawContours(original_image, [box], 0, color, 2)
            
            # Draw box number
            cv2.putText(original_image, f'#{i+1}', (int(x), int(y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            results.append({
                "box_id": i + 1,
                "position": {"x": round(x, 2), "y": round(y, 2)},
                "orientation": round(angle, 2),
                "vertices": box.tolist()
            })
        
        return original_image, results

    except Exception as e:
        return None, {"error": str(e)}

def main():
    # Load an image instead of using webcam
    image_path = 'WhatsApp Image 2024-12-21 at 17.06.04_5f28055a.jpg'  # Replace with your image file path
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Unable to load image")
        return
    
    # Process the image
    processed_image, results = detect_boxes(image)
    
    if processed_image is not None:
        # Show the processed image with boxes
        cv2.imshow("Processed Image", processed_image)
        
        # Print the detection results
        print(f"Detection Results: {results}")
        
        # Wait until any key is pressed, then close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
