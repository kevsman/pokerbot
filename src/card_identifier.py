import cv2
import numpy as np
import pytesseract
import time
import os
import logging

# Set up logger
logger = logging.getLogger(__name__)


class CardIdentifier:
    """Class for identifying cards from images"""
    
    def __init__(self, debug_mode=True):
        """Initialize the card identifier"""
        self.debug_mode = debug_mode
        # Set tesseract path explicitly if it's not in PATH
        # Uncomment and set if needed
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
    def identify_card(self, card_img):
        """
        Identify the rank and suit of a card from its image using a more robust approach.
        
        Args:
            card_img (numpy.ndarray): Image of a single card.
            
        Returns:
            tuple: (rank, suit) of the card, or (None, None) if not identified.
        """
        try:
            # Resize the card image for more consistent processing
            h, w = card_img.shape[:2]
            resized = cv2.resize(card_img, (100, int(100 * h/w)))
            
            # Extract the top-left corner where the rank and suit are usually displayed
            # Increased corner_h from 40 to 50 to ensure we capture the full rank character
            corner_h, corner_w = min(50, resized.shape[0]//2), min(30, resized.shape[1]//2)
            corner = resized[0:corner_h, 0:corner_w]
            
            # Save corner image for debugging
            if self.debug_mode:
                debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug')
                os.makedirs(debug_dir, exist_ok=True)
                corner_path = os.path.join(debug_dir, f'card_corner_{int(time.time())}.png')
                cv2.imwrite(corner_path, corner)
                logger.info(f"[CARD DEBUG] Saved card corner image to {corner_path}")
            
            # Enhanced color analysis for suits
            # Convert to multiple color spaces for better analysis
            hsv_corner = cv2.cvtColor(corner, cv2.COLOR_BGR2HSV)
            
            # Improved red detection (for hearts and diamonds) with wider thresholds
            # In HSV, red is at both ends of the hue spectrum
            lower_red1 = np.array([0, 70, 70])     # Lowered saturation and value thresholds
            upper_red1 = np.array([15, 255, 255])  # Increased hue range to catch more red variations
            lower_red2 = np.array([160, 70, 70])   # Lowered saturation and value thresholds
            upper_red2 = np.array([180, 255, 255])
            
            red_mask1 = cv2.inRange(hsv_corner, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv_corner, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # Improved black detection (for clubs and spades) with adjusted thresholds
            # In HSV, black has low V (value/brightness)
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 150, 100])  # Increased saturation and value thresholds
            black_mask = cv2.inRange(hsv_corner, lower_black, upper_black)
            
            # Count red and black pixels
            red_pixels = cv2.countNonZero(red_mask)
            black_pixels = cv2.countNonZero(black_mask)
            
            # Save masks for debugging
            if self.debug_mode:
                debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug')
                cv2.imwrite(os.path.join(debug_dir, f'red_mask_{int(time.time())}.png'), red_mask)
                cv2.imwrite(os.path.join(debug_dir, f'black_mask_{int(time.time())}.png'), black_mask)
                logger.info(f"[CARD DEBUG] Red pixels: {red_pixels}, Black pixels: {black_pixels}")
            
            # Determine if card is red or black with improved comparison
            # Added pixel density threshold relative to the image size
            total_pixels = corner.shape[0] * corner.shape[1]
            min_pixel_threshold = max(10, total_pixels * 0.05)  # At least 5% of pixels or 10 pixels
            is_red = red_pixels > black_pixels and red_pixels > min_pixel_threshold
            
            # Enhanced OCR for rank detection with multiple preprocessing approaches
            # Try multiple preprocessing methods and choose the best result
            ocr_results = []
            confidence_scores = []
            
            # Original method
            gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
            
            # Additional preprocessing - try different thresholds
            _, binary1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            _, binary2 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            
            # Add contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
            enhanced = clahe.apply(gray)
            _, binary3 = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Dilate to connect broken parts of characters
            kernel = np.ones((2,2), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=1)
            binary1 = cv2.dilate(binary1, kernel, iterations=1)
            binary2 = cv2.dilate(binary2, kernel, iterations=1)
            binary3 = cv2.dilate(binary3, kernel, iterations=1)
            
            # Save thresholded image for debugging
            if self.debug_mode:
                debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug')
                cv2.imwrite(os.path.join(debug_dir, f'rank_thresh_{int(time.time())}.png'), thresh)
                cv2.imwrite(os.path.join(debug_dir, f'rank_binary1_{int(time.time())}.png'), binary1)
                cv2.imwrite(os.path.join(debug_dir, f'rank_binary2_{int(time.time())}.png'), binary2)
                cv2.imwrite(os.path.join(debug_dir, f'rank_binary3_{int(time.time())}.png'), binary3)
            
            # Use tesseract with specific configurations for card rank detection
            # Try multiple PSM modes for better results
            
            # PSM 8 - Single word
            rank_config1 = r'--psm 8 -c tessedit_char_whitelist=23456789TJQKA10 --oem 3'
            # PSM 10 - Single character
            rank_config2 = r'--psm 10 -c tessedit_char_whitelist=23456789TJQKA10 --oem 3'
            # PSM 6 - Single block of text
            rank_config3 = r'--psm 6 -c tessedit_char_whitelist=23456789TJQKA10 --oem 3'
            
            # Try all combinations of image preprocessing and OCR configs
            for img, name in [(thresh, 'thresh'), (binary1, 'binary1'), (binary2, 'binary2'), (binary3, 'binary3')]:
                for config, config_name in [(rank_config1, 'config1'), (rank_config2, 'config2'), (rank_config3, 'config3')]:
                    try:
                        data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
                        if len(data['text']) > 0 and len(data['conf']) > 0:
                            # Filter out empty results
                            valid_indices = [i for i, txt in enumerate(data['text']) if txt.strip()]
                            if valid_indices:
                                text = data['text'][valid_indices[0]].strip()
                                conf = float(data['conf'][valid_indices[0]])
                                if text and conf > 0:  # Only add non-empty results with positive confidence
                                    ocr_results.append(text)
                                    confidence_scores.append(conf)
                                    if self.debug_mode:
                                        logger.info(f"[CARD DEBUG] OCR {name}+{config_name}: '{text}' (conf: {conf})")
                    except Exception as e:
                        logger.debug(f"OCR error with {name}+{config_name}: {e}")
            
            # Improved mapping of OCR results to card ranks with more common misidentifications
            rank_map = {
                '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', 
                '8': '8', '9': '9', '1': '10', 'T': '10', 'J': 'J', 
                'Q': 'Q', 'K': 'K', 'A': 'A', 'l': '1', 'I': '1',
                'O': '10', 'o': '10', '0': '10', 'L': 'J', 'Z': '2',
                't': '10', 'i': '1', '!': '1', '[': 'J', ']': 'J',
                'S': '5', 'B': '8', 'G': '6', 'g': '9',
                'U': 'J', 'V': 'A', 'Y': 'A', 'W': 'M',
                'P': 'F', 'F': 'P', 'D': '0', 'H': '4',
                'R': 'K', 'X': 'K', 'N': '7', 'M': 'W',
                'C': '0', 'E': '3', '?': '2', '#': '4',
                '*': 'A', '+': 'A', '<': 'K', '>': 'K'
            }
            
            # Process OCR text for rank with improved logic
            rank = None
            best_confidence = -1
            detected_text = ""
            
            # If we have OCR results, use the one with highest confidence
            if ocr_results and confidence_scores:
                best_idx = np.argmax(confidence_scores)
                detected_text = ocr_results[best_idx]
                best_confidence = confidence_scores[best_idx]
            
            if detected_text:
                logger.info(f"[CARD DEBUG] Best OCR text: '{detected_text}' (confidence: {best_confidence})")
                
                # Clean up OCR results
                detected_text = ''.join(c for c in detected_text if c.isalnum())
                
                # If we found something like '10' directly or its variations
                if detected_text in ['10', '1O', 'IO', 'To', 'T0', 'TO', 'TQ', 'I0', 'TD', '1D', 'TP', '1P']:
                    rank = '10'
                else:
                    # Try to match the first character to a rank
                    if detected_text and detected_text[0] in rank_map:
                        rank = rank_map[detected_text[0]]
                    
                    # Special case for 10, check if any 2-character substring matches
                    if detected_text and len(detected_text) > 1:
                        for i in range(len(detected_text) - 1):
                            if detected_text[i:i+2] in ['10', '1O', 'IO', 'To', 'T0', 'TO', 'TQ', 'I0', 'TD', '1D', 'TP', '1P']:
                                rank = '10'
                                break
            
            # If OCR failed or had low confidence, try shape analysis methods
            if not rank or best_confidence < 40:  # Only rely on OCR if confidence is good
                # Focus on the very top-left where rank is typically located
                rank_roi = corner[0:min(25, corner_h), 0:min(25, corner_w)]
                
                # Try multiple shape detection methods
                if self._check_for_A_shape(rank_roi):
                    rank = 'A'
                elif self._check_for_K_shape(rank_roi):
                    rank = 'K'
                elif self._check_for_Q_shape(rank_roi):
                    rank = 'Q'
                elif self._check_for_J_shape(rank_roi):
                    rank = 'J'
                elif self._check_for_9_shape(rank_roi):
                    rank = '9'
            
                # Use a pattern-based approach as a last resort
                if not rank:
                    rank = self._rank_by_pattern_matching(thresh)
            
            # Determine suit based on improved shape analysis
            suit = None
            
            # For red cards (hearts and diamonds)
            if is_red:
                # Apply morphological operations to better detect shape features
                kernel = np.ones((2,2), np.uint8)
                red_processed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
                
                # Find contours in the red parts
                contours, _ = cv2.findContours(red_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Heart detection: hearts typically have a "V" shape at the bottom
                    heart_score = self._detect_heart_shape(contours, red_processed.shape)
                    
                    # Diamond detection: diamonds typically have sharp corners forming a rhombus
                    diamond_score = self._detect_diamond_shape(contours, red_processed.shape)
                    
                    logger.info(f"[CARD DEBUG] Heart score: {heart_score}, Diamond score: {diamond_score}")
                    
                    # Determine suit based on which score is higher
                    if heart_score > diamond_score:
                        suit = 'h'
                    else:
                        suit = 'd'
                else:
                    # If contour analysis fails, default to diamond as it's more common in online poker
                    suit = 'd'
            else:
                # For black cards (clubs and spades)
                kernel = np.ones((2,2), np.uint8)
                black_processed = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(black_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Spade detection: spades typically have a triangular shape at the top
                    spade_score = self._detect_spade_shape(contours, black_processed.shape)
                    
                    # Club detection: clubs typically have three circular lobes
                    club_score = self._detect_club_shape(contours, black_processed.shape)
                    
                    logger.info(f"[CARD DEBUG] Spade score: {spade_score}, Club score: {club_score}")
                    
                    # Determine suit based on which score is higher
                    if spade_score > club_score:
                        suit = 's'
                    else:
                        suit = 'c'
                else:
                    # If contour analysis fails, default to spade as it's more common
                    suit = 's'
            
            # Try a second approach for suits if the first wasn't definitive
            if suit and (heart_score < 10 and diamond_score < 10) or (spade_score < 10 and club_score < 10):
                # Try to use shape detection directly on the corner image
                suit_region = corner[corner_h//3:, :]  # Focus on the lower part of the corner for suit
                suit2 = self._detect_suit_by_template(suit_region, is_red)
                if suit2:
                    suit = suit2
            
            # If we couldn't detect either rank or suit, return None
            if not rank or not suit:
                logger.info(f"[CARD DEBUG] Failed to identify card: rank={rank}, suit={suit}")
                return None, None
                
            logger.info(f"[CARD DEBUG] Successfully identified card: {rank}{suit}, is_red={is_red}")
            return rank, suit
            
        except Exception as e:
            logger.exception(f"Error identifying card: {e}")
            return None, None
            
    def _detect_suit_by_template(self, img, is_red):
        """Try to detect the suit using template-based approach"""
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Create binary image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Skip if no significant contours found
        if not contours:
            return None
            
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Skip if contour is too small
        if cv2.contourArea(largest_contour) < 10:
            return None
            
        # Calculate shape features
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w)/h if h > 0 else 0
        area = cv2.contourArea(largest_contour)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area if hull_area > 0 else 0
        
        # Analyze shape characteristics
        if is_red:
            # For red suits: heart vs diamond
            if aspect_ratio < 0.9:  # Hearts are typically taller than wide
                if solidity < 0.8:  # Hearts have concavity at the top
                    return 'h'
                else:
                    return 'd'  # Diamonds tend to be more solid
            else:
                return 'd'  # Diamond is more likely for wider shapes
        else:
            # For black suits: club vs spade
            if solidity > 0.7:
                # Clubs tend to be more solid/compact
                return 'c'
            else:
                # Spades tend to have a triangular shape
                return 's'
                
        return None
            
    def _rank_by_pattern_matching(self, img):
        """Try to identify a card rank by its pattern of white pixels."""
        h, w = img.shape[:2]
        
        # Count white pixels in different regions
        top_left = cv2.countNonZero(img[0:h//3, 0:w//3])
        top_middle = cv2.countNonZero(img[0:h//3, w//3:2*w//3])
        top_right = cv2.countNonZero(img[0:h//3, 2*w//3:w])
        
        middle_left = cv2.countNonZero(img[h//3:2*h//3, 0:w//3])
        middle_middle = cv2.countNonZero(img[h//3:2*h//3, w//3:2*w//3])
        middle_right = cv2.countNonZero(img[h//3:2*h//3, 2*w//3:w])
        
        bottom_left = cv2.countNonZero(img[2*h//3:h, 0:w//3])
        bottom_middle = cv2.countNonZero(img[2*h//3:h, w//3:2*w//3])
        bottom_right = cv2.countNonZero(img[2*h//3:h, 2*w//3:w])
        
        # Very simplified pattern recognition based on the distribution of white pixels
        total_pixels = top_left + top_middle + top_right + middle_left + middle_middle + middle_right + bottom_left + bottom_middle + bottom_right
        if total_pixels < 10:  # Too few pixels to classify
            return None
            
        # A very rough estimation based on typical card rank shapes
        if top_middle > top_left and top_middle > top_right and bottom_middle > bottom_left and bottom_middle > bottom_right:
            # Central vertical line pattern suggests 1 or T
            return '10'
        elif top_left > top_right and bottom_right > bottom_left:
            # Diagonal pattern may suggest K
            return 'K'
        elif top_left > 0 and top_right > 0 and bottom_middle > 0:
            # U shape pattern suggests J
            return 'J'
        elif (top_left > 0 and top_right > 0 and 
              middle_left > 0 and middle_right > 0 and 
              bottom_left > 0 and bottom_right > 0):
            # O shape pattern suggests Q or 0
            return 'Q'
        elif top_middle > 0 and middle_middle > 0 and bottom_left > 0 and bottom_right > 0:
            # Top vertical with bottom horizontals suggests A
            return 'A'
        else:
            return None
            
    def _check_for_9_shape(self, img):
        """Check for '9' characteristic shape (circle with tail)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        
        h, w = binary.shape
        if h < 10 or w < 10:
            return False
            
        # 9 typically has a circular top part
        top_half = binary[0:h//2, :]
        contours, _ = cv2.findContours(top_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        has_circle = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5:  # Ignore tiny contours
                continue
                
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            if circularity > 0.4:  # Circle-like shape
                has_circle = True
                break
                
        # Check for tail in bottom right
        bottom_right = binary[h//2:h, w//2:w]
        bottom_right_pixels = cv2.countNonZero(bottom_right)
        
        # Characteristic of '9': circle top and tail at bottom
        if has_circle and bottom_right_pixels > 2:
            return True
            
        return False
        
    def _check_for_A_shape(self, img):
        """Check for 'A' characteristic shape (a peak with diverging lines)"""
        # Convert to binary
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        
        # Look for white pixels forming a triangular distribution
        h, w = binary.shape
        if h < 10 or w < 10:  # Too small to analyze
            return False
            
        # Check for a concentration of white pixels in the upper middle 
        # and diverging pattern toward bottom
        upper_mid = binary[0:h//2, w//4:3*w//4]
        upper_mid_pixels = cv2.countNonZero(upper_mid)
        
        bottom_left = binary[h//2:h, 0:w//2]
        bottom_left_pixels = cv2.countNonZero(bottom_left)
        
        bottom_right = binary[h//2:h, w//2:w]
        bottom_right_pixels = cv2.countNonZero(bottom_right)
        
        # Characteristic of 'A': strong presence in upper middle and both bottom corners
        if (upper_mid_pixels > 5 and
            bottom_left_pixels > 5 and
            bottom_right_pixels > 5):
            return True
        return False
        
    def _check_for_K_shape(self, img):
        """Check for 'K' characteristic shape (vertical line with diagonal branches)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        
        h, w = binary.shape
        if h < 10 or w < 10:
            return False
            
        # 'K' typically has a strong vertical line on the left
        left_col = binary[:, 0:w//4]
        left_pixels = cv2.countNonZero(left_col)
        
        # And diagonal elements from middle to right
        mid_right = binary[:, w//3:w]
        mid_right_pixels = cv2.countNonZero(mid_right)
        
        # Characteristic of 'K': strong left vertical and diagonal components
        if (left_pixels > h/2 and mid_right_pixels > 5):
            return True
        return False
        
    def _check_for_Q_shape(self, img):
        """Check for 'Q' characteristic shape (circular with a tail)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        
        h, w = binary.shape
        if h < 10 or w < 10:
            return False
        
        # 'Q' typically has a circular pattern in the top and middle
        top_and_middle = binary[0:3*h//4, :]
        
        # Q has a distinct diagonal tail specifically in the bottom right
        bottom_right = binary[2*h//3:h, 2*w//3:w]
        bottom_right_pixels = cv2.countNonZero(bottom_right)
        
        # Check middle left and right for O-like shape (should have pixels on both sides)
        middle_left = binary[h//4:3*h//4, 0:w//3]
        middle_right = binary[h//4:3*h//4, 2*w//3:w]
        middle_left_pixels = cv2.countNonZero(middle_left)
        middle_right_pixels = cv2.countNonZero(middle_right)
        
        # Check middle center for empty space (the hole in the "O" shape)
        middle_center = binary[h//4:3*h//4, w//3:2*w//3]
        middle_center_pixels = cv2.countNonZero(middle_center)
        middle_center_empty = middle_center_pixels < (middle_center.shape[0] * middle_center.shape[1]) * 0.5
        
        # Q-specific: Should have a gap in the middle center (open circle) 
        # and pixels on both left and right sides
        has_side_pixels = middle_left_pixels > 3 and middle_right_pixels > 3
        
        # Check for circle-like contour in top and middle area
        contours, _ = cv2.findContours(top_and_middle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            # Check if contour is approximately circular
            area = cv2.contourArea(cnt)
            if area < 5:  # Ignore tiny contours
                continue
                
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # 'Q' shape: circular top and middle, plus diagonal tail in bottom right
            # Must have side pixels on both left and right in middle region to be "O"-like
            # Q has a specific trait - middle_center should be more empty than K
            # Add a stronger check for bottom right tail which is distinctive for Q
            if (circularity > 0.5 and 
                has_side_pixels and 
                middle_center_empty and
                bottom_right_pixels > 2):
                return True
                
        return False
        
    def _check_for_J_shape(self, img):
        """Check for 'J' characteristic shape (vertical with hook at bottom)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        
        h, w = binary.shape
        if h < 10 or w < 10:
            return False
            
        # 'J' typically has vertical component in the middle
        middle_col = binary[:, w//4:3*w//4]
        middle_pixels = cv2.countNonZero(middle_col)
        
        # And a hook at the bottom left
        bottom_left = binary[3*h//4:h, 0:w//2]
        bottom_left_pixels = cv2.countNonZero(bottom_left)
        
        # Characteristic of 'J': middle vertical and bottom left hook
        if (middle_pixels > h/3 and bottom_left_pixels > 3):
            return True
        return False
        
    def _detect_heart_shape(self, contours, shape):
        """Return a score indicating how heart-like the contours are"""
        score = 0
        h, w = shape
        
        # Enhanced heart characteristics: 
        # 1. Two circular bumps at top
        # 2. Pointed bottom
        # 3. Symmetrical along vertical axis
        
        for cnt in contours:
            if len(cnt) < 5:
                continue
                
            try:
                # Get bounding box
                x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
                
                # Check aspect ratio - hearts are typically taller than wide
                if w_cnt > 0 and 0.7 < h_cnt/w_cnt < 2.0:
                    score += 5
                
                # Check for pointed bottom - safely extract y coordinates
                cnt_reshaped = cnt.reshape(-1, 2)  # Reshape to 2D array of [x,y] coordinates
                y_coords = cnt_reshaped[:, 1]      # Get all y coordinates
                bottom_y = np.max(y_coords)
                
                # Find points near the bottom
                bottom_points = cnt_reshaped[y_coords >= bottom_y - 3]
                
                # A pointed bottom will have relatively few points at the max y-coordinate
                if len(bottom_points) < 5:
                    score += 8
                
                # Check for two bumps at top using convexity defects
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                cnt_area = cv2.contourArea(cnt)
                
                # Hearts have concavities (the dip between the two lobes and bottom point)
                # so contour area is significantly less than hull area
                if cnt_area > 0 and hull_area / cnt_area > 1.2:
                    score += 12
                    
                # Check for symmetry along vertical axis
                # First, determine the vertical midline
                left_x = np.min(cnt_reshaped[:, 0])
                right_x = np.max(cnt_reshaped[:, 0])
                mid_x = (left_x + right_x) / 2
                
                # Split points into left and right of midline
                left_points = cnt_reshaped[cnt_reshaped[:, 0] < mid_x]
                right_points = cnt_reshaped[cnt_reshaped[:, 0] >= mid_x]
                
                # For hearts, the distribution should be roughly symmetric
                if left_points.shape[0] > 0 and right_points.shape[0] > 0:
                    left_ratio = left_points.shape[0] / cnt_reshaped.shape[0]
                    # Hearts should have approximately equal distribution
                    if 0.4 <= left_ratio <= 0.6:
                        score += 10
                
            except Exception as e:
                logger.debug(f"Error in heart shape detection: {e}")
                
        return score
        
    def _detect_diamond_shape(self, contours, shape):
        """Return a score indicating how diamond-like the contours are"""
        score = 0
        h, w = shape
        
        # Enhanced diamond characteristics: 
        # 1. Four corners with similar angles (rhombus)
        # 2. High symmetry both horizontally and vertically
        # 3. More compact shape compared to heart
        
        for cnt in contours:
            if len(cnt) < 5:
                continue
                
            try:
                # Check for polygon approximation with 4 points (square/diamond)
                epsilon = 0.04 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                if len(approx) == 4:
                    score += 15  # Strong indicator of diamond
                elif 3 <= len(approx) <= 6:  # Allow some tolerance
                    score += 5
                
                # Get bounding rect
                x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
                
                # Check if width/height ratio is close to 1 (diamond is typically symmetric)
                if w_cnt > 0 and 0.7 < h_cnt/w_cnt < 1.4:
                    score += 8
                
                # Check convexity - diamonds are convex shapes
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                cnt_area = cv2.contourArea(cnt)
                
                # A perfect diamond should have area close to its convex hull
                if cnt_area > 0 and hull_area > 0:
                    solidity = cnt_area / hull_area
                    if solidity > 0.8:  # Diamond is very solid (no concavities)
                        score += 10
                        
                # Check for symmetry - diamonds are symmetrical
                cnt_reshaped = cnt.reshape(-1, 2)
                if len(cnt_reshaped) > 0:
                    # Horizontal symmetry
                    left_x = np.min(cnt_reshaped[:, 0])
                    right_x = np.max(cnt_reshaped[:, 0])
                    mid_x = (left_x + right_x) / 2
                    
                    left_points = cnt_reshaped[cnt_reshaped[:, 0] < mid_x]
                    right_points = cnt_reshaped[cnt_reshaped[:, 0] >= mid_x]
                    
                    if left_points.shape[0] > 0 and right_points.shape[0] > 0:
                        left_ratio = left_points.shape[0] / cnt_reshaped.shape[0]
                        if 0.4 <= left_ratio <= 0.6:  # Roughly equal distribution
                            score += 5
                            
                    # Vertical symmetry
                    top_y = np.min(cnt_reshaped[:, 1])
                    bottom_y = np.max(cnt_reshaped[:, 1])
                    mid_y = (top_y + bottom_y) / 2
                    
                    top_points = cnt_reshaped[cnt_reshaped[:, 1] < mid_y]
                    bottom_points = cnt_reshaped[cnt_reshaped[:, 1] >= mid_y]
                    
                    if top_points.shape[0] > 0 and bottom_points.shape[0] > 0:
                        top_ratio = top_points.shape[0] / cnt_reshaped.shape[0]
                        if 0.4 <= top_ratio <= 0.6:  # Roughly equal distribution
                            score += 5
                
            except Exception as e:
                logger.debug(f"Error in diamond shape detection: {e}")
                
        return score
        
    def _detect_spade_shape(self, contours, shape):
        """Return a score indicating how spade-like the contours are"""
        score = 0
        h, w = shape
        
        # Enhanced spade characteristics:
        # 1. Triangular top
        # 2. Small stem at bottom
        # 3. Symmetrical along vertical axis
        
        for cnt in contours:
            if len(cnt) < 5:
                continue
                
            try:
                # Reshape contour to 2D array for easier indexing
                cnt_reshaped = cnt.reshape(-1, 2)
                
                # Check overall aspect ratio - spades are typically taller than wide
                x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
                if w_cnt > 0 and h_cnt/w_cnt > 1.2:
                    score += 5
                
                # Get the topmost point - find the minimum y-coordinate
                y_coords = cnt_reshaped[:, 1]
                topmost_idx = np.argmin(y_coords)
                
                # Check for triangular top
                # Get points in the top half
                top_half_indices = np.where(cnt_reshaped[:, 1] < (y_coords[0] + h)//2)[0]
                if len(top_half_indices) >= 3:
                    top_half = cnt_reshaped[top_half_indices]
                    
                    # Try to fit a triangle to top half points
                    top_hull = cv2.convexHull(top_half.reshape(-1, 1, 2))
                    epsilon = 0.04 * cv2.arcLength(top_hull, True)
                    approx = cv2.approxPolyDP(top_hull, epsilon, True)
                    
                    if len(approx) == 3:
                        score += 12  # Strong indicator of spade
                    elif len(approx) == 4:
                        score += 5   # Could still be a spade
                        
                # Check for narrow stem at bottom
                # Get points in the bottom third
                bottom_indices = np.where(cnt_reshaped[:, 1] > 2*h//3)[0]
                if len(bottom_indices) > 0:
                    bottom_part = cnt_reshaped[bottom_indices]
                    
                    if len(bottom_part) > 0:
                        # Calculate width of bottom part
                        x_coords = bottom_part[:, 0]
                        bottom_width = np.max(x_coords) - np.min(x_coords)
                        
                        # Spades have a distinctive narrow stem at the bottom
                        if bottom_width < w//2:
                            score += 8
                        
                # Check for symmetry along vertical axis (spades are symmetric)
                left_x = np.min(cnt_reshaped[:, 0])
                right_x = np.max(cnt_reshaped[:, 0])
                mid_x = (left_x + right_x) / 2
                
                # Split points into left and right of midline
                left_points = cnt_reshaped[cnt_reshaped[:, 0] < mid_x]
                right_points = cnt_reshaped[cnt_reshaped[:, 0] >= mid_x]
                
                if left_points.shape[0] > 0 and right_points.shape[0] > 0:
                    left_ratio = left_points.shape[0] / cnt_reshaped.shape[0]
                    # Spades should have approximately equal distribution
                    if 0.4 <= left_ratio <= 0.6:
                        score += 8
                        
            except Exception as e:
                logger.debug(f"Error in spade shape detection: {e}")
                    
        return score
        
    def _detect_club_shape(self, contours, shape):
        """Return a score indicating how club-like the contours are"""
        score = 0
        h, w = shape
        
        # Enhanced club characteristics:
        # 1. Multiple circular lobes (typically 3)
        # 2. Small stem at bottom
        # 3. Distinctive triple-circle pattern
        
        # First, check if there are multiple distinct contours (club lobes)
        if len(contours) >= 2:
            score += 10  # Multiple contours are a strong club indicator
            
        # Check for 3 distinctly separated regions in the upper part
        if len(contours) == 3:
            score += 5  # Perfect club match
            
        for cnt in contours:
            if len(cnt) < 5:
                continue
                
            try:
                # Reshape contour to 2D array
                cnt_reshaped = cnt.reshape(-1, 2)
                
                # Check if contour is approximately circular (club lobes are circular)
                area = cv2.contourArea(cnt)
                if area < 5:  # Ignore tiny contours
                    continue
                    
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                if circularity > 0.6:
                    score += 12  # Strong indicator for club's circular lobes
                    
                # Check for compactness - clubs tend to be compact overall
                x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
                extent = float(area) / (w_cnt * h_cnt) if (w_cnt * h_cnt) > 0 else 0
                
                # Check if the contour fills a significant portion of its bounding rectangle
                if 0.4 < extent < 0.8:  # Club's multiple lobes create a medium extent
                    score += 5
                    
                # Check for small stem at bottom
                bottom_indices = np.where(cnt_reshaped[:, 1] > 2*h//3)[0]
                if len(bottom_indices) > 0:
                    bottom_part = cnt_reshaped[bottom_indices]
                    
                    if len(bottom_part) > 0:
                        # Calculate width of bottom part
                        x_coords = bottom_part[:, 0]
                        bottom_width = np.max(x_coords) - np.min(x_coords)
                        
                        if bottom_width < w//3:
                            score += 8  # Narrow stem is typical for clubs
                            
                # Check for three-lobe pattern using moments
                moments = cv2.moments(cnt)
                if moments['m00'] != 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                    
                    # For a club, points should distribute evenly around the center of mass
                    # in the upper portion (the three lobes)
                    upper_points = cnt_reshaped[cnt_reshaped[:, 1] < cy]
                    
                    if len(upper_points) > 0:
                        # Calculate distances from each point to centroid
                        dists = np.sqrt((upper_points[:, 0] - cx)**2 + (upper_points[:, 1] - cy)**2)
                        mean_dist = np.mean(dists)
                        std_dist = np.std(dists)
                        
                        # For clubs, there should be a consistent pattern of distances
                        # (the three lobes are roughly equidistant from center)
                        if std_dist / mean_dist < 0.5:  # Low variation in distances
                            score += 8
                            
            except Exception as e:
                logger.debug(f"Error in club shape detection: {e}")
                    
        return score

    def preprocess_for_ocr(self, img):
        """Preprocess an image for better OCR results"""
        # Convert to grayscale if it's not already
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Apply some morphological operations to clean up the image
        kernel = np.ones((2,2), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        
        return morph