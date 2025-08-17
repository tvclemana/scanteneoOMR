import argparse, glob, os, sys, csv, json, random
import numpy as np
import cv2 as cv

# ===================== TEMPLATE GENERATION =====================
def create_answer_sheet_template(width=2480, height=3508, questions=30, options=5):
    """
    Create a blank OMR answer sheet template.
    Returns the template image and bubble positions.
    """
    # Create white background
    template = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Template parameters
    margin = 140
    question_start_y = margin + 200
    row_height = 90
    bubble_radius = 22
    gap_between_options = 70
    question_number_width = 60
    cols = 2
    questions_per_col = questions // cols
    
    # Colors
    black = (0, 0, 0)
    gray = (128, 128, 128)
    
    bubble_positions = {}
    
    # Draw title
    cv.putText(template, "OMR ANSWER SHEET", (width//2 - 200, margin), 
               cv.FONT_HERSHEY_SIMPLEX, 1.5, black, 3)
    
    # Draw instructions
    cv.putText(template, "Fill in the circle completely for your answer", (width//2 - 300, margin + 80), 
               cv.FONT_HERSHEY_SIMPLEX, 0.8, gray, 2)
    
    # Draw questions and bubbles
    for col in range(cols):
        col_x = margin + 120 + col * (width//2 - 60)
        
        for row in range(questions_per_col):
            question_num = col * questions_per_col + row + 1
            y = question_start_y + row * row_height
            
            # Draw question number
            cv.putText(template, str(question_num), (col_x, y + 8), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, black, 2)
            
            # Draw option letters and bubbles
            if question_num not in bubble_positions:
                bubble_positions[question_num] = {}
            
            for opt_idx, option in enumerate(['A', 'B', 'C', 'D', 'E']):
                x = col_x + question_number_width + opt_idx * gap_between_options
                
                # Draw option letter
                cv.putText(template, option, (x - 8, y + 8), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, gray, 1)
                
                # Draw bubble (empty circle)
                cv.circle(template, (x, y), bubble_radius, black, 2)
                
                # Store bubble position
                bubble_positions[question_num][option] = (x, y, bubble_radius)
    
    return template, bubble_positions

def create_filled_answer_sheet(template, bubble_positions, answers):
    """
    Create a filled answer sheet based on the provided answers.
    answers: dict mapping question number to option (A, B, C, D, E)
    """
    filled_sheet = template.copy()
    
    # Fill in the bubbles based on answers
    for question_num, answer in answers.items():
        if question_num in bubble_positions and answer in bubble_positions[question_num]:
            x, y, r = bubble_positions[question_num][answer]
            # Fill the circle completely
            cv.circle(filled_sheet, (x, y), r, (0, 0, 0), -1)
    
    return filled_sheet

# ===================== TEMPLATE-BASED DETECTION =====================
def detect_template_structure(image):
    """
    Detect the template structure (question numbers, option letters) to guide bubble detection.
    Returns the detected template parameters.
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    
    h, w = gray.shape
    
    # Estimate template parameters based on image dimensions
    estimated_margin = int(min(w, h) * 0.05)  # 5% of smaller dimension
    estimated_row_height = int(h * 0.025)  # 2.5% of height
    estimated_bubble_radius = int(min(w, h) * 0.01)  # 1% of smaller dimension
    
    # Use morphological operations to find text-like regions
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dilated = cv.dilate(gray, kernel, iterations=1)
    eroded = cv.erode(dilated, kernel, iterations=1)
    
    # Find contours that might be text
    contours, _ = cv.findContours(eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size to find text regions
    text_regions = []
    for contour in contours:
        x, y, w_contour, h_contour = cv.boundingRect(contour)
        area = w_contour * h_contour
        
        # Text regions are typically small rectangles
        if 50 < area < 500 and 5 < w_contour < 50 and 5 < h_contour < 50:
            text_regions.append((x, y, w_contour, h_contour))
    
    # Sort text regions by y-coordinate to identify rows
    if text_regions:
        text_regions.sort(key=lambda r: r[1])
        
        # Group into rows
        rows = []
        current_row = [text_regions[0]]
        row_tolerance = estimated_row_height // 2
        
        for region in text_regions[1:]:
            if abs(region[1] - current_row[0][1]) <= row_tolerance:
                current_row.append(region)
            else:
                # Sort current row by x-coordinate
                current_row.sort(key=lambda r: r[0])
                rows.append(current_row)
                current_row = [region]
        
        if current_row:
            current_row.sort(key=lambda r: r[0])
            rows.append(current_row)
    else:
        rows = []
    
    # Estimate template parameters
    template_params = {
        'margin': estimated_margin,
        'row_height': estimated_row_height,
        'bubble_radius': estimated_bubble_radius,
        'text_regions': text_regions,
        'rows': rows,
        'image_width': w,
        'image_height': h
    }
    
    return template_params

def detect_bubbles_with_template(image, template_params):
    """
    Detect bubbles using the template structure as a guide.
    """
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    
    h, w = gray.shape
    
    # Use template parameters to predict bubble locations
    margin = template_params['margin']
    row_height = template_params['row_height']
    bubble_radius = template_params['bubble_radius']
    
    # Predict bubble positions based on template structure
    predicted_bubbles = []
    
    # Estimate question layout (2 columns, 15 questions per column)
    questions_per_col = 15
    col_width = w // 2
    
    for col in range(2):
        col_x = margin + 120 + col * col_width
        
        for row in range(questions_per_col):
            y = margin + 200 + row * row_height
            
            # Predict 5 options per question
            for opt_idx in range(5):
                x = col_x + 60 + opt_idx * 70  # 60 for question number, 70 between options
                
                # Check if this region actually contains a bubble
                # Look for circular patterns in the predicted location
                roi_x1 = max(0, x - bubble_radius - 5)
                roi_y1 = max(0, y - bubble_radius - 5)
                roi_x2 = min(w, x + bubble_radius + 5)
                roi_y2 = min(h, y + bubble_radius + 5)
                
                roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
                
                if roi.size > 0:
                    # Use Hough Circle Transform to detect actual circles
                    try:
                        circles = cv.HoughCircles(
                            roi,
                            cv.HOUGH_GRADIENT,
                            dp=1,
                            minDist=bubble_radius * 2,
                            param1=50,
                            param2=30,
                            minRadius=bubble_radius - 5,
                            maxRadius=bubble_radius + 5
                        )
                        
                        if circles is not None:
                            # Found a circle, add to detected bubbles
                            for circle in circles[0]:
                                cx, cy, r = circle
                                # Convert back to full image coordinates
                                full_x = int(roi_x1 + cx)
                                full_y = int(roi_y1 + cy)
                                predicted_bubbles.append((full_x, full_y, int(r)))
                    except Exception as e:
                        # Skip this region if there's an error
                        continue
    
    return predicted_bubbles

def organize_bubbles_into_questions(bubbles, template_params):
    """
    Organize detected bubbles into question-option structure.
    """
    if not bubbles:
        return {}
    
    # Sort bubbles by y-coordinate to group by rows
    bubbles.sort(key=lambda b: b[1])
    
    # Group bubbles by rows
    row_tolerance = template_params['row_height'] // 2
    rows = []
    current_row = [bubbles[0]]
    
    for bubble in bubbles[1:]:
        if abs(bubble[1] - current_row[0][1]) <= row_tolerance:
            current_row.append(bubble)
        else:
            # Sort current row by x-coordinate
            current_row.sort(key=lambda b: b[0])
            rows.append(current_row)
            current_row = [bubble]
    
    if current_row:
        current_row.sort(key=lambda b: b[0])
        rows.append(current_row)
    
    # Organize into question-option structure
    bubble_positions = {}
    question_num = 1
    
    for row in rows:
        if len(row) >= 3:  # Need at least 3 options to be valid
            # Map options A, B, C, D, E to the bubbles in this row
            options = ['A', 'B', 'C', 'D', 'E']
            for i, (x, y, r) in enumerate(row[:5]):  # Take first 5 bubbles
                if question_num not in bubble_positions:
                    bubble_positions[question_num] = {}
                bubble_positions[question_num][options[i]] = (x, y, r)
            question_num += 1
    
    return bubble_positions

# ===================== SCORING =====================
def fill_ratio(binary_img, cx, cy, r):
    """Calculate the fill ratio of a bubble."""
    h, w = binary_img.shape[:2]
    r = max(5, r)
    
    x0, y0 = max(0, cx - r), max(0, cy - r)
    x1, y1 = min(w-1, cx + r), min(h-1, cy + r)
    
    roi = binary_img[y0:y1+1, x0:x1+1]
    if roi.size == 0:
        return 0.0
    
    # Create circular mask
    Y, X = np.ogrid[:roi.shape[0], :roi.shape[1]]
    mask = (X - (cx - x0))**2 + (Y - (cy - y0))**2 <= r*r
    
    # Count dark pixels (filled marks)
    dark_pixels = np.count_nonzero(roi[mask] < 128)
    total_pixels = mask.sum()
    
    return dark_pixels / total_pixels if total_pixels > 0 else 0.0

def decide_answer(ratios, min_fill=0.3):
    """Decide which option is selected based on fill ratios."""
    if not ratios:
        return None, 0.0
    
    # Find the option with the highest fill ratio
    best_option = max(ratios.items(), key=lambda x: x[1])
    
    if best_option[1] >= min_fill:
        return best_option[0], best_option[1]
    else:
        return None, best_option[1]

def score_sheet(image, bubble_positions, answer_key):
    """Score the OMR sheet against the answer key."""
    # Convert to grayscale and threshold
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply threshold to get binary image
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    results = {}
    correct_count = 0
    
    for question_num, options in bubble_positions.items():
        if question_num not in results:
            results[question_num] = {}
        
        # Calculate fill ratios for each option
        ratios = {}
        for option, (x, y, r) in options.items():
            ratio = fill_ratio(binary, x, y, r)
            ratios[option] = ratio
        
        # Decide the selected answer
        selected, confidence = decide_answer(ratios)
        
        # Check if correct
        correct_answer = answer_key.get(question_num)
        is_correct = selected == correct_answer
        
        if is_correct:
            correct_count += 1
        
        results[question_num] = {
            'selected': selected,
            'confidence': confidence,
            'correct': correct_answer,
            'is_correct': is_correct,
            'ratios': ratios
        }
    
    return results, correct_count

# ===================== MAIN FUNCTIONS =====================
def generate_mock_sheets(output_dir="."):
    """Generate mock answer sheets for testing."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate blank template
    template, bubble_positions = create_answer_sheet_template()
    template_path = os.path.join(output_dir, "omr_blank_template.png")
    cv.imwrite(template_path, template)
    
    # Generate random answers
    answers = {}
    for q in range(1, 31):
        answers[q] = random.choice(['A', 'B', 'C', 'D', 'E'])
    
    # Generate filled sheet
    filled_sheet = create_filled_answer_sheet(template, bubble_positions, answers)
    filled_path = os.path.join(output_dir, "omr_filled_test.png")
    cv.imwrite(filled_path, filled_sheet)
    
    # Save answer key
    answer_key_path = os.path.join(output_dir, "omr_answer_key.csv")
    with open(answer_key_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['question', 'answer'])
        for q in sorted(answers.keys()):
            writer.writerow([q, answers[q]])
    
    # Save bubble positions for reference
    positions_path = os.path.join(output_dir, "omr_bubble_positions.json")
    with open(positions_path, 'w') as f:
        json.dump(bubble_positions, f, indent=2)
    
    print(f"Generated mock sheets in {output_dir}:")
    print(f"  - Blank template: {template_path}")
    print(f"  - Filled test sheet: {filled_path}")
    print(f"  - Answer key: {answer_key_path}")
    print(f"  - Bubble positions: {positions_path}")
    
    return template_path, filled_path, answer_key_path, positions_path

def process_omr_sheet(image_path, answer_key_path=None):
    """Process an OMR sheet and return results."""
    # Read image
    image = cv.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Detect template structure
    print("Detecting template structure...")
    template_params = detect_template_structure(image)
    
    # Detect bubbles using template guidance
    print("Detecting bubbles...")
    detected_bubbles = detect_bubbles_with_template(image, template_params)
    print(f"Detected {len(detected_bubbles)} bubbles")
    
    # Organize bubbles into questions
    print("Organizing bubbles into questions...")
    bubble_positions = organize_bubbles_into_questions(detected_bubbles, template_params)
    print(f"Organized into {len(bubble_positions)} questions")
    
    # Load answer key if provided
    answer_key = {}
    if answer_key_path and os.path.exists(answer_key_path):
        with open(answer_key_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                answer_key[int(row['question'])] = row['answer']
    
    # Score the sheet
    if answer_key:
        print("Scoring sheet...")
        results, correct_count = score_sheet(image, bubble_positions, answer_key)
        
        # Calculate score
        total_questions = len(bubble_positions)
        score_percentage = (correct_count / total_questions * 100) if total_questions > 0 else 0
        
        print(f"Score: {correct_count}/{total_questions} ({score_percentage:.1f}%)")
        
        return {
            'bubble_positions': bubble_positions,
            'results': results,
            'score': correct_count,
            'total': total_questions,
            'percentage': score_percentage
        }
    else:
        return {
            'bubble_positions': bubble_positions,
            'message': 'No answer key provided for scoring'
        }

# ===================== CLI =====================
def main():
    parser = argparse.ArgumentParser(description="OMR Sheet Generator and Processor")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate mock OMR sheets')
    gen_parser.add_argument('--output-dir', default='.', help='Output directory for generated sheets')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process an OMR sheet')
    process_parser.add_argument('image_path', help='Path to the OMR sheet image')
    process_parser.add_argument('--answer-key', help='Path to the answer key CSV file')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        generate_mock_sheets(args.output_dir)
    elif args.command == 'process':
        try:
            results = process_omr_sheet(args.image_path, args.answer_key)
            print("\nProcessing Results:")
            print(json.dumps(results, indent=2))
        except Exception as e:
            print(f"Error processing sheet: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
