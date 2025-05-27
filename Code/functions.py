import numpy as np
import cv2
from PyQt5.QtGui import QPixmap, QImage


def dft_analysis(img):
    """
    Enhanced DFT analysis with better visualization and normalization
    
    Parameters:
    - img: Input color image
    
    Returns:
    - mag_img: Magnitude spectrum visualization
    - phase_img: Phase spectrum visualization  
    - interpretation: Text explanation of the DFT results
    - dft_components: Dictionary containing raw DFT components for later use
    """
    # Convert to grayscale for DFT
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Optimal DFT size (for improved performance)
    rows, cols = gray.shape
    optimal_rows = cv2.getOptimalDFTSize(rows)
    optimal_cols = cv2.getOptimalDFTSize(cols)
    
    # Create zero-padded image for optimal FFT performance
    padded = np.zeros((optimal_rows, optimal_cols), np.float32)
    padded[:rows, :cols] = gray
    
    # Perform DFT and shift to center
    dft = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Calculate magnitude spectrum
    mag = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    
    # Apply log scale for better visualization (log(1+mag))
    mag_log = np.log1p(mag)
    mag_normalized = cv2.normalize(mag_log, None, 0, 255, cv2.NORM_MINMAX)
    
    # Calculate phase spectrum
    phase = np.arctan2(dft_shift[:, :, 1], dft_shift[:, :, 0])
    phase_normalized = (phase + np.pi) * (255 / (2 * np.pi))
    
    # Create enhanced visualizations with colormaps
    mag_img = cv2.applyColorMap(np.uint8(mag_normalized), cv2.COLORMAP_JET)
    phase_img = cv2.applyColorMap(np.uint8(phase_normalized), cv2.COLORMAP_HSV)
    
    # Calculate interpretive metrics
    mean_magnitude = np.mean(mag_log)
    std_magnitude = np.std(mag_log)
    features_count = np.sum(mag_log > (mean_magnitude + 2*std_magnitude))
    
    # Generate interpretation text
    interpretation = f"""DFT Analysis Interpretation:
    
Magnitude Spectrum:
- The bright central point represents the DC component (average brightness)
- Bright spots away from center indicate strong frequencies/patterns
- Horizontal/vertical lines suggest edges in vertical/horizontal directions
- Mean magnitude: {mean_magnitude:.2f}
- Feature count: {features_count}
    
Phase Spectrum:
- Shows spatial relationships between frequency components
- Uniform colors indicate consistent patterns
- Mixed colors suggest complex spatial relationships
- Essential for accurate image reconstruction
    
This image appears to have {"high" if features_count > 200 else "moderate" if features_count > 50 else "low"} 
frequency content, suggesting {"many detailed features" if features_count > 200 else "some patterns" if features_count > 50 else "mostly uniform regions"}.
"""
    
    # Store components for later use in interpolation
    dft_components = {
        'dft_shift': dft_shift,
        'magnitude': mag,
        'phase': phase,
        'shape': (rows, cols),
        'optimal_shape': (optimal_rows, optimal_cols)
    }
    
    return mag_img, phase_img, interpretation, dft_components


def combine_magnitude_phase(magnitude_components, phase_components):
    """
    Combines magnitude from one image with phase from another image
    to demonstrate the importance of phase information
    
    Parameters:
    - magnitude_components: DFT components from first image
    - phase_components: DFT components from second image
    
    Returns:
    - reconstructed_img: Reconstructed image using magnitude from first and phase from second
    - interpretation: Text explanation of the result
    """
    # Check if the optimal shapes match
    mag_shape = magnitude_components['optimal_shape']
    phase_shape = phase_components['optimal_shape']
    
    # Use the smaller dimensions for processing
    optimal_rows = min(mag_shape[0], phase_shape[0])
    optimal_cols = min(mag_shape[1], phase_shape[1])
    orig_rows = min(magnitude_components['shape'][0], phase_components['shape'][0])
    orig_cols = min(magnitude_components['shape'][1], phase_components['shape'][1])
    
    # Get magnitude and phase
    magnitude = magnitude_components['magnitude'][:optimal_rows, :optimal_cols]
    phase = phase_components['phase'][:optimal_rows, :optimal_cols]
    
    # Create complex DFT with magnitude from img1 and phase from img2
    real = magnitude * np.cos(phase)
    imaginary = magnitude * np.sin(phase)
    
    combined_dft = np.zeros((optimal_rows, optimal_cols, 2), dtype=np.float32)
    combined_dft[:, :, 0] = real
    combined_dft[:, :, 1] = imaginary
    
    # Shift back to original position
    combined_dft_shift = np.fft.ifftshift(combined_dft)
    
    # Inverse DFT to get reconstructed image
    reconstructed = cv2.idft(combined_dft_shift)
    reconstructed = cv2.magnitude(reconstructed[:, :, 0], reconstructed[:, :, 1])
    
    # Normalize and crop to original size
    reconstructed = cv2.normalize(reconstructed, None, 0, 255, cv2.NORM_MINMAX)
    reconstructed = reconstructed[:orig_rows, :orig_cols]
    
    # Convert to 8-bit for display
    reconstructed_img = np.uint8(reconstructed)
    
    # Apply color map for visualization
    reconstructed_color = cv2.applyColorMap(reconstructed_img, cv2.COLORMAP_BONE)
    
    # Create interpretation text
    interpretation = """Phase-Magnitude Interpolation Analysis:
    
This image combines:
- Magnitude spectrum from one image (overall frequency strengths)
- Phase spectrum from the other image (spatial relationships)

Key observations:
- The reconstructed image demonstrates that phase information carries 
  most of the structural content of the image
- Magnitude contributes to intensity distribution and contrast
- This hybrid image shows how complementary frequency components 
  from different sources can create new visualizations

What you're seeing:
- The structural features primarily come from the phase-contributing image
- The texture and intensity patterns come from the magnitude-contributing image
- This illustrates why phase is typically more important for image recognition

This technique helps understand:
- The relative importance of phase vs. magnitude information in images
- How our visual system processes and interprets complex signals
- The principles behind frequency-domain image processing
"""
    
    return reconstructed_color, interpretation


def reconstruct_custom_image(dft_components, use_magnitude=True, use_phase=True, 
                            mag_weight=1.0, phase_weight=1.0, colormap=cv2.COLORMAP_BONE):
    """
    Reconstruct an image with custom settings for magnitude and phase
    
    Parameters:
    - dft_components: Dictionary containing DFT components
    - use_magnitude: Whether to use magnitude (True) or constant value (False)
    - use_phase: Whether to use phase (True) or zero phase (False)
    - mag_weight: Weight for magnitude (0.0-1.0)
    - phase_weight: Weight for phase (0.0-1.0)
    - colormap: OpenCV colormap for visualization
    
    Returns:
    - reconstructed_img: Reconstructed image
    - interpretation: Text explanation of the reconstruction
    """
    # Get DFT components
    dft_shift = dft_components['dft_shift']
    magnitude = dft_components['magnitude']
    phase = dft_components['phase']
    orig_shape = dft_components['shape']
    optimal_shape = dft_components['optimal_shape']
    
    # Prepare magnitude
    if not use_magnitude:
        # Use constant magnitude (1.0)
        magnitude = np.ones_like(magnitude)
    else:
        # Apply weight to magnitude
        if mag_weight != 1.0:
            # Apply exponential weighting to modify magnitude distribution
            magnitude = magnitude ** mag_weight
    
    # Prepare phase
    if not use_phase:
        # Use zero phase
        phase = np.zeros_like(phase)
    else:
        # Apply weight to phase
        if phase_weight != 1.0:
            # Reduce phase weight by interpolating between original phase and zero
            phase = phase * phase_weight
    
    # Create complex DFT components
    real = magnitude * np.cos(phase)
    imaginary = magnitude * np.sin(phase)
    
    combined_dft = np.zeros((optimal_shape[0], optimal_shape[1], 2), dtype=np.float32)
    combined_dft[:, :, 0] = real
    combined_dft[:, :, 1] = imaginary
    
    # Shift back to original position
    combined_dft_shift = np.fft.ifftshift(combined_dft)
    
    # Inverse DFT to get reconstructed image
    reconstructed = cv2.idft(combined_dft_shift)
    reconstructed = cv2.magnitude(reconstructed[:, :, 0], reconstructed[:, :, 1])
    
    # Normalize and crop to original size
    reconstructed = cv2.normalize(reconstructed, None, 0, 255, cv2.NORM_MINMAX)
    reconstructed = reconstructed[:orig_shape[0], :orig_shape[1]]
    
    # Convert to 8-bit for display
    reconstructed_img = np.uint8(reconstructed)
    
    # Apply color map for visualization
    if colormap is not None:
        reconstructed_color = cv2.applyColorMap(reconstructed_img, colormap)
    else:
        reconstructed_color = cv2.cvtColor(reconstructed_img, cv2.COLOR_GRAY2BGR)
    
    # Create interpretation text
    mag_str = f"{int(mag_weight*100)}% magnitude" if use_magnitude else "No magnitude (constant)"
    phase_str = f"{int(phase_weight*100)}% phase" if use_phase else "No phase (zero)"
    
    interpretation = f"""Image Reconstruction Analysis:

This image was reconstructed using:
- {mag_str}
- {phase_str}

Key observations:
- {'Phase information provides the structural details' if use_phase else 'Without phase, structural details are lost'}
- {'Magnitude affects contrast and energy distribution' if use_magnitude else 'Without magnitude, only structural outlines remain'}
- {'Both components contribute to a complete reconstruction' if use_magnitude and use_phase else 'This partial reconstruction shows the importance of the missing component'}

What you're seeing:
- {'A complete image with all frequency information' if use_magnitude and use_phase and mag_weight == 1.0 and phase_weight == 1.0 
   else 'A partial reconstruction demonstrating the importance of specific DFT components'}
- {'Reduced magnitude emphasizes dominant structures' if use_magnitude and mag_weight < 1.0 else ''}
- {'Reduced phase blurs structural boundaries' if use_phase and phase_weight < 1.0 else ''}

This visualization demonstrates:
- {'How phase and magnitude independently contribute to image formation' if use_magnitude != use_phase 
   else 'How adjusting the balance between phase and magnitude affects image quality'}
"""
    
    return reconstructed_color, interpretation

def np_to_pixmap(img_np):
    """Convert numpy image to QPixmap for display"""
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    bytes_per_line = ch * w
    return QPixmap.fromImage(QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888))

def calculate_reconstruction_metrics(original_img, reconstructed_img):
    """Calculate metrics for comparing original and reconstructed images"""
    # Ensure images are grayscale
    if len(original_img.shape) > 2:
        original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original_img.copy()
        
    if len(reconstructed_img.shape) > 2:
        reconstructed_gray = cv2.cvtColor(reconstructed_img, cv2.COLOR_BGR2GRAY)
    else:
        reconstructed_gray = reconstructed_img.copy()
    
    # Ensure sizes match
    h, w = min(original_gray.shape[0], reconstructed_gray.shape[0]), min(original_gray.shape[1], reconstructed_gray.shape[1])
    original_gray = original_gray[:h, :w]
    reconstructed_gray = reconstructed_gray[:h, :w]
    
    # Calculate MSE (Mean Squared Error)
    mse = np.mean((original_gray.astype(np.float32) - reconstructed_gray.astype(np.float32)) ** 2)
    
    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10((255 ** 2) / mse)
    
    # Calculate SSIM (Structural Similarity Index)
    try:
        ssim = cv2.compareSSIM(original_gray, reconstructed_gray)
    except:
        ssim = 0  # Default if SSIM calculation fails
    
    return {
        'MSE': mse,
        'PSNR': psnr,
        'SSIM': ssim
    }
'''
def optimized_harris_corner_detector(img, block_size=3, k_param=0.04, threshold_ratio=0.01):
    """
    Custom Harris corner detector implementation without using OpenCV's built-in Harris function
    
    Parameters:
    - img: Input color image
    - block_size: Size of neighborhood considered for corner detection
    - k_param: Harris detector free parameter in the equation
    - threshold_ratio: Ratio of maximum response value to use as detection threshold
    
    Returns:
    - corner_img: Image with corners marked
    - response_visualization: Visualization of Harris response
    - num_corners: Number of corners detected
    - interpretation: Text explanation of the Harris detection results
    """
    # Convert to grayscale for processing
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Convert to float32 for calculations
    gray = np.float32(gray)
    
    # Get image dimensions
    height, width = gray.shape
    
    # Step 1: Calculate image derivatives (gradient) using Sobel
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # Step 2: Calculate gradient products for the covariance matrix
    dx2 = dx * dx
    dy2 = dy * dy
    dxy = dx * dy
    
    # Step 3: Apply Gaussian blur to the gradient products (approximation of summing over block)
    gaussian_sigma = block_size / 3  # Approximate conversion from block size to sigma
    dx2_blur = cv2.GaussianBlur(dx2, (block_size*2+1, block_size*2+1), gaussian_sigma)
    dy2_blur = cv2.GaussianBlur(dy2, (block_size*2+1, block_size*2+1), gaussian_sigma)
    dxy_blur = cv2.GaussianBlur(dxy, (block_size*2+1, block_size*2+1), gaussian_sigma)
    
    # Step 4: Calculate Harris response (R = det(M) - k * trace(M)^2)
    det_M = dx2_blur * dy2_blur - dxy_blur * dxy_blur
    trace_M = dx2_blur + dy2_blur
    harris_response = det_M - k_param * (trace_M ** 2)
    
    # Step 5: Apply thresholding
    max_response = np.max(harris_response)
    threshold = threshold_ratio * max_response
    
    # Step 6: Non-maximum suppression
    # Dilate to get local maxima
    kernel = np.ones((block_size*2+1, block_size*2+1), np.uint8)
    harris_dilated = cv2.dilate(harris_response, kernel)
    
    # Get coordinates of corners (where response equals dilated response and is above threshold)
    corner_coords = []
    corner_map = np.zeros_like(gray, dtype=np.uint8)
    for y in range(block_size, height-block_size):
        for x in range(block_size, width-block_size):
            if harris_response[y, x] > threshold and harris_response[y, x] == harris_dilated[y, x]:
                corner_coords.append((x, y))
                corner_map[y, x] = 255
    
    # Step 7: Draw corners on the original image
    corner_img = img.copy() if len(img.shape) > 2 else cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    
    for x, y in corner_coords:
        cv2.circle(corner_img, (x, y), 3, (0, 255, 0), -1)  # Green circles for corners
        cv2.circle(corner_img, (x, y), 5, (0, 0, 255), 1)   # Red circle outlines
    
    # Create a colored visualization of Harris response
    response_normalized = cv2.normalize(harris_response, None, 0, 255, cv2.NORM_MINMAX)
    response_colored = cv2.applyColorMap(np.uint8(response_normalized), cv2.COLORMAP_JET)
    
    # Blend the corner map with response visualization
    alpha = 0.7
    response_visualization = cv2.addWeighted(
        response_colored, alpha, 
        cv2.cvtColor(corner_map, cv2.COLOR_GRAY2BGR), 1-alpha, 
        0
    )
    
    # Calculate the density of corners (corners per 1000 pixels)
    num_pixels = height * width
    num_corners = len(corner_coords)
    corner_density = (num_corners * 1000) / num_pixels
    
    # Generate interpretation text
    interpretation = f"""Harris Corner Detection Results:
    
Parameters:
- Block Size: {block_size}
- k Parameter: {k_param}
- Threshold: {threshold_ratio*100:.1f}% of maximum response

Detection Results:
- Total corners detected: {num_corners}
- Corner density: {corner_density:.2f} corners per 1000 pixels
- Image dimensions: {width} × {height} pixels

Interpretation:
- {"High" if corner_density > 1.0 else "Moderate" if corner_density > 0.5 else "Low"} corner density detected
- This suggests {"a detailed image with many features" if corner_density > 1.0 else 
                "an image with some distinctive features" if corner_density > 0.5 else 
                "an image with few distinctive features"}
- Corner detection is {"very sensitive" if threshold_ratio < 0.01 else 
                      "moderately sensitive" if threshold_ratio < 0.05 else 
                      "not very sensitive"} with current threshold

Harris Detection Theory:
- The algorithm calculates image gradients in x and y directions
- It constructs a 2×2 structure tensor (M) for each pixel
- Corners have large variations in all directions, resulting in high response values
- Harris response R = det(M) - k×trace(M)²
- Corners are identified as local maxima of the response function above threshold
"""
    
    return corner_img, response_visualization, num_corners, interpretation
    
'''


def harris_corner_detector(img, block_size=3, k_param=0.04, threshold_ratio=0.01):
    """
    Improved Harris corner detector with accurate corner marking.
    
    Args:
        img: Input color image (BGR format)
        block_size: Neighborhood size (odd number >=3)
        k_param: Sensitivity factor (typically 0.04-0.06)
        threshold_ratio: Ratio of max response for thresholding (0.01 = 1%)
        
    Returns:
        corners_gray: Grayscale corner response image
        marked_img: Original image with corners marked in red
        corner_points: List of (x,y) corner coordinates
    """
    # Convert to grayscale and float32
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # 1. Compute gradients
    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # 2. Compute elements of the structure tensor
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy
    
    # 3. Apply Gaussian blur (instead of box filter)
    kernel_size = (block_size, block_size)
    Sx2 = cv2.GaussianBlur(Ix2, kernel_size, 0)
    Sy2 = cv2.GaussianBlur(Iy2, kernel_size, 0)
    Sxy = cv2.GaussianBlur(Ixy, kernel_size, 0)
    
    # 4. Compute corner response
    det = Sx2 * Sy2 - Sxy**2
    trace = Sx2 + Sy2
    R = det - k_param * (trace**2)
    
    # 5. Normalize and threshold response
    R_norm = cv2.normalize(R, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    threshold = threshold_ratio * R.max()
    
    # 6. Find corner coordinates
    corner_coords = np.argwhere(R > threshold)
    corner_points = [(x, y) for y, x in corner_coords]  # Convert to (x,y) format
    
    # 7. Non-maximum suppression
    R[R <= threshold] = 0
    R_dilated = cv2.dilate(R, None)
    local_max = (R == R_dilated)
    corners = np.zeros_like(gray, dtype=np.uint8)
    corners[local_max] = 255
    
    # 8. Create marked image
    marked_img = img.copy()
    for y, x in corner_coords:
        cv2.circle(marked_img, (x, y), 3, (0, 0, 255), -1)  # Draw red circles
    
    return R_norm, marked_img, corner_points,""

# Example usage
if __name__ == "__main__":
    img = cv2.imread('chessboard.png')
    if img is not None:
        gray, marked, corners = harris_corner_detector(img)
        
        print(f"Detected {len(corners)} corners")
        
        cv2.imshow('Corner Response', gray)
        cv2.imshow('Marked Corners', marked)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Image not loaded")