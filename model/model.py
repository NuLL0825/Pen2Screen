import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import system
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import torch
import json
import torch.nn.functional as F
import shutil
import re

from flask import Flask, request, jsonify


def visualize_image(title, image, cmap='gray'):
    """Helper function to display an image with a title."""
    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.imshow(image, cmap=cmap)
    plt.axis("off")
    plt.show()

def preprocess_color_image(image):
    """
    Preprocess a color image by converting it to grayscale with emphasis on color ink visibility.
    """
    b, g, r = cv2.split(image)
    grayscale = cv2.addWeighted(r, 0.33, g, 0.33, 0)
    grayscale = cv2.addWeighted(grayscale, 1.0, b, 0.33, 0)
    return grayscale

def merge_bounding_boxes(contours, min_distance=10):
    """Merge bounding boxes that are close to each other."""
    merged_boxes = []
    
    # Sort contours based on the x-coordinate (left to right)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        merged = False
        
        for i, (mx, my, mw, mh) in enumerate(merged_boxes):
            # Check if the bounding boxes are close enough to merge
            if abs(x - mx) < min_distance and abs(y - my) < min_distance:
                # Merge bounding boxes
                merged_boxes[i] = (min(x, mx), min(y, my), max(x + w, mx + mw) - min(x, mx), max(y + h, my + mh) - min(y, my))
                merged = True
                break
        
        if not merged:
            merged_boxes.append((x, y, w, h))
    
    return merged_boxes

# Save segmented regions (characters or chars)
def save_segmented_regions(image, regions, output_dir, prefix="char", count=0, savedidx=0):
    os.makedirs(output_dir, exist_ok=True)
    newidx = savedidx
    for idx, (x1, y1, x2, y2) in enumerate(regions):
        roi = image[y1:y2, x1:x2]
        output_path = os.path.join(output_dir, f"{prefix}_{count:03d}_{idx + 1:03d}.png")
        cv2.imwrite(output_path, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
        print(f"Saved: {output_path}")
        newidx += 1
    return newidx

        
def save_image_to_folder(image_path, destination_folder, filename, prefix, count, idx):
    """
    Save the given image to the specified destination folder with a custom naming convention.

    :param image_path: Path to the image to be saved.
    :param destination_folder: Folder where the image should be saved.
    :param filename: Original filename of the image.
    :param prefix: Prefix to use in the naming convention.
    :param count: Count to use in the naming convention.
    :param idx: Index for the specific image (e.g., for multiple segmented parts).
    """
    os.makedirs(destination_folder, exist_ok=True)  # Ensure the folder exists

    # Construct the custom filename based on the provided naming convention
    new_filename = f"{prefix}_{count:03d}_{idx + 1:03d}.png"
    destination_path = os.path.join(destination_folder, new_filename)

    try:
        # Read and save the image
        image = Image.open(image_path)
        image.save(destination_path)
        print(f"Saved {filename} as {new_filename} to {destination_folder}")
    except Exception as e:
        print(f"Error saving {filename}: {e}")


def clear_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Iterate through all files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # Remove the file
                if os.path.isfile(file_path):
                    os.remove(file_path)
                # If it's a directory, remove it and its contents
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error deleting file {filename}: {e}")
        print(f"All files in {folder_path} have been deleted.")
    else:
        print(f"The folder {folder_path} does not exist.")
        
def show_segmented_characters(image, regions):
    """
    Display all the segmented characters in a grid format.
    
    :param image: Original grayscale or color image.
    :param regions: List of bounding boxes [(x1, y1, x2, y2)] for the characters.
    """
    print(f"Total Segments Found: {len(regions)}")
    plt.figure(figsize=(15, 5))  # Adjust figure size for better visibility
    
    # Loop through each region
    for idx, (x1, y1, x2, y2) in enumerate(regions, 1):
        # Extract the region of interest (ROI)
        roi = image[y1:y2, x1:x2]
        
        # Plot the ROI in a grid
        plt.subplot(1, len(regions), idx)  # Create a subplot for each segment
        plt.imshow(roi, cmap="gray")
        plt.title(f"Char {idx}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()
       

def segment_expression(image_path, output_dir, merge_distance_x, merge_distance_y = 5, iter = 7, binarize_value=60, count=0):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return

    # Step 1: Convert to grayscale for color inks
    grayscale = preprocess_color_image(image)
    # visualize_image("Grayscale Conversion for Color Inks", grayscale)

    # Step 2: Noise Reduction using Gaussian Blur
    denoised = cv2.GaussianBlur(grayscale, (5, 5), 0)
    # visualize_image("After Noise Reduction (Gaussian Blur)", denoised)

    # Step 3: Sharpen the Image
    sharpening_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, sharpening_kernel)
    # visualize_image("After Sharpening", sharpened)

    # Step 4: Binary Thresholding
    _, binary = cv2.threshold(sharpened, binarize_value, 255, cv2.THRESH_BINARY_INV)
    # visualize_image("Binary Threshold Image", binary)

    # Step 5: Dilation to Merge Components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (merge_distance_x, merge_distance_y))  # Larger kernel for merging
    dilated = cv2.dilate(binary, kernel, iterations = iter)
    # visualize_image("Dilated Binary Image", dilated)

    # Step 6: Find Contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 7: Merge Bounding Boxes
    merged_contours = merge_bounding_boxes(contours, min_distance=20)

    # Create a copy of the original image to visualize results
    segmented_image = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)

    # List to store bounding box coordinates
    regions = []

    # Loop through each merged bounding box
    for (x, y, w, h) in merged_contours:
        # Ignore very small components
        if w > 10 and h > 10:
            # Draw a rectangle around each component
            cv2.rectangle(segmented_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Save bounding box coordinates
            regions.append((x, y, x + w, y + h))

    # Visualize the segmented image with bounding boxes
    # visualize_image("Segmented Image with Bounding Boxes", segmented_image)
    try:
        savedidx = newidx
    except:
        savedidx = 0

    # Save the segmented regions
    newidx = save_segmented_regions(image, regions, output_dir,"char", count, savedidx)


    # Return bounding box coordinates and ROIs (optional)
    return regions
def segment_expression_to_images(
    image_path, 
    merge_distance_x, 
    merge_distance_y=5, 
    iter=7, 
    binarize_value=60
):
    """
    Segments an image into regions, processes each region, 
    and returns them as a list of image arrays.

    Args:
        image_path (str): Path to the input image.
        merge_distance_x (int): Horizontal merge distance for dilation.
        merge_distance_y (int): Vertical merge distance for dilation.
        iter (int): Number of iterations for dilation.
        binarize_value (int): Thresholding value for binarization.

    Returns:
        List of segmented image arrays.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Step 1: Convert to grayscale for color inks
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Noise Reduction using Gaussian Blur
    denoised = cv2.GaussianBlur(grayscale, (5, 5), 0)

    # Step 3: Sharpen the Image
    sharpening_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, sharpening_kernel)

    # Step 4: Binary Thresholding
    _, binary = cv2.threshold(sharpened, binarize_value, 255, cv2.THRESH_BINARY_INV)

    # Step 5: Dilation to Merge Components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (merge_distance_x, merge_distance_y))
    dilated = cv2.dilate(binary, kernel, iterations=iter)

    # Step 6: Find Contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 7: Merge Bounding Boxes
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, x + w, y + h))

    # Sort bounding boxes (left to right, top to bottom)
    bounding_boxes = sorted(bounding_boxes, key=lambda b: (b[0], b[1]))

    # List to store processed images
    processed_images = []

    for (x1, y1, x2, y2) in bounding_boxes:
        # Ignore very small components
        if (x2 - x1) > 10 and (y2 - y1) > 10:
            # Crop the region   
            cropped_image = grayscale[y1:y2, x1:x2]

            # Resize and normalize for model compatibility
            processed = cv2.resize(cropped_image, (32, 32), interpolation=cv2.INTER_AREA)
            pil_image = Image.fromarray(processed)
            processed_images.append(pil_image)
    
    return processed_images

def model_character_recognition(image_path, latex_index_path, device):
    # Load LaTeX token mapping
    with open(latex_index_path) as f:
        ltx_index = json.load(f)

    tokenizer = system.LatexTokenizer(ltx_index['symbol_to_index'])
    post_processor = system.LatexPostProcessor(ltx_index['index_to_symbol'])
        
    num_classes = len(ltx_index['symbol_to_index'])

    model = system.CRNN(imgH=32, nc=1, nclass=num_classes, nh=256).to(device)

    model.load_state_dict(torch.load("fine_tuned_CRNN.pth" , weights_only=True, map_location=torch.device('cpu'))) # walang gpu :(

    model.eval()
    
    image = Image.open(image_path)

    # # Visualize the original image
    # plt.figure(figsize=(6, 4))
    # plt.imshow(image, cmap='gray')  # Use cmap='gray' for grayscale images
    # plt.title("Original Image")
    # plt.axis('off')  # Hide axes
    # plt.show()

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 128)),  # Match model input size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to match training
    ])
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Predict
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_tensor = input_tensor.to(device)
    output = model(input_tensor)

    # Apply softmax to get probabilities
    probabilities = F.softmax(output, dim=2)  # Shape: [T, B, num_classes]

    # Get predicted class indices and their probabilities
    predicted_indices = torch.argmax(probabilities, dim=2)  # Shape: [T, B]
    predicted_confidences, _ = torch.max(probabilities, dim=2)  # Shape: [T, B]

    # Decode predictions and print confidence levels
    for t, (index, confidence) in enumerate(zip(predicted_indices[:, 0], predicted_confidences[:, 0])):
        decoded_char = post_processor.index_to_symbol.get(str(index.item()), '<UNK>')
        if decoded_char == '<PAD>':
            break  # Stop processing further tokens
        print(f"Character: {decoded_char}, Confidence: {confidence.item():.4f}")
        
def model_character_recognition_alter(image, ltx_index, device): # model_character_recognition but with direct input (not path)
    # Load LaTeX token mapping (debug)
    # with open(ltx_index) as f:
    #     ltx_index = json.load(f)
    # image = Image.open(image)    

    tokenizer = system.LatexTokenizer(ltx_index['symbol_to_index'])
    post_processor = system.LatexPostProcessor(ltx_index['index_to_symbol'])
        
    num_classes = len(ltx_index['symbol_to_index'])

    model = system.CRNN(imgH=32, nc=1, nclass=num_classes, nh=256).to(device)

    model.load_state_dict(torch.load("fine_tuned_CRNN.pth" , weights_only=True, map_location=torch.device('cpu'))) # walang gpu :(

    model.eval()

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 128)),  # Match model input size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to match training
    ])
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Predict
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_tensor = input_tensor.to(device)
    output = model(input_tensor)

    # Apply softmax to get probabilities
    probabilities = F.softmax(output, dim=2)  # Shape: [T, B, num_classes]

    # Get predicted class indices and their probabilities
    predicted_indices = torch.argmax(probabilities, dim=2)  # Shape: [T, B]
    predicted_confidences, _ = torch.max(probabilities, dim=2)  # Shape: [T, B]

    # # Decode predictions and print confidence levels
    for t, (index, confidence) in enumerate(zip(predicted_indices[:, 0], predicted_confidences[:, 0])):
        decoded_char = post_processor.index_to_symbol.get(str(index.item()), '<UNK>')
        if decoded_char == '<PAD>':
            break  # Stop processing further tokens
        predicted_confidences = confidence.item()
        predicted_indices = post_processor.index_to_symbol.get(str(index.item()), '<UNK>')
    return predicted_indices, predicted_confidences
def is_latex_match(image, latex_mapping, device):
    """
    Check if the image contains any LaTeX symbols from the mapping.
    Returns True if a match is found, else False.
    """
    # Placeholder: Implement your character recognition logic here
    keys = ["\\lim_", "=", "\\csc", "\\sin", "\\cos", "\\tan", "\\log", "\\cot", "\\cos", "\\sec"]
    latex_key = latex_mapping['symbol_to_index']
    char, confidence = detect_symbols(image, latex_mapping, device)
    
    detected_symbols = char  # Function to detect symbols in the image
    for symbol in latex_key.keys():
        if symbol == detected_symbols and symbol in keys:
            return True, confidence, char
    return False, confidence, char

def detect_symbols(image, latex_mapping, device):
    """
    Placeholder function to detect symbols in an image.
    This needs to be implemented based on your model or processing pipeline.
    """
    # This is where you would integrate your text recognition or symbol detection logic.
    # For now, we assume a list of detected symbols for the example.
    char, confidence = model_character_recognition_alter(image, latex_mapping, device)
    return char, confidence  # Example detected symbols
def augment_images_from_strings(strings, art_count, output_dir="artificial_chars", image_size=(200, 200), font_size=60):
    """
    Generate images for an array of strings and save them as individual files.

    Args:
        strings (list of str): List of strings to generate images for.
        output_dir (str): Directory where images will be saved.
        image_size (tuple): Size of each image (width, height).
        font_size (int): Font size for the text.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load a font
    try:
        font = ImageFont.truetype("arial.ttf", size=font_size)  # Adjust font path if needed
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font if arial.ttf isn't found

    # Iterate over the strings and generate an image for each
    for count, text in enumerate(strings):
        lowercase_text = text.lower()
        
        # Create a blank image with a white background
        image = Image.new("RGB", image_size, color="white")
        draw = ImageDraw.Draw(image)

        # Calculate text position to center it using textbbox
        bbox = draw.textbbox((0, 0), lowercase_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)

        # Draw the text on the image
        draw.text(text_position, text, fill="black", font=font)

        # Save the image
        output_path = os.path.join(output_dir, f"letter_image_{art_count:03d}_{count + 1:03d}.png")
        image.save(output_path)

        print(f"Image saved to {output_path}")
        
def check_pattern(array, pattern):
    # Join the array into a single string
    joined_array = ''.join(array)
    
    # Check if any pattern appears as a substring in the joined string
    for pat in pattern:
        if pat in joined_array:
            return True
    return False
def write_latex(strings, output_dir, prefix="char", count=0, savedidx=0):
    for idx, string in enumerate(strings):
        # Create a file name using the index of the string
        file_name = f"{output_dir}/{prefix}_{count:03d}_{idx + 1:03d}.txt"

        # Open the file in write mode and write the string to it
        with open(file_name, "w") as file:
            file.write(string)
        
        print(f"Created file: {file_name} with content: {string}")
def process_images_in_group(segmented_chars_folder, segmented_groups_folder, latex_index_path, device): # second process
    """
    Process each image in the segmented_groups folder, check for LaTeX matches,
    move matched images to segmented_chars, and then perform character segmentation
    with higher sensitivity.
    """
    with open(latex_index_path, 'r') as f:
        latex_mapping = json.load(f)
    count = 0
    art_count = 0
    prefix = "char"
    keys = ["lim", "csc", "sin", "cos", "tan", "log", "cot", "cos", "sec"]
    alt_keys = ["c0s", "l0g", "c0t", "l1m", "s1n", "lin"]
    trans_keys = str.maketrans({'0': 'o', '1': 'i'})
    separator = ''
    
    
    for filename in os.listdir(segmented_groups_folder):
        image_path = os.path.join(segmented_groups_folder, filename)
        temp_arr = []
        temp = ""
        final_arr = []
        
        
        saved_char = []
        combined_char = ""
        
        if os.path.isfile(image_path):
            # image = cv2.imread(image_path)
            image = Image.open(image_path)
            count += 1
            
            
            is_latex, confidence, char = is_latex_match(image, latex_mapping, device)
            
            
            # Check if the image contains any LaTeX symbols from latex_mapping
            if is_latex and confidence >= 0.95:
                # Move the image to segmented_chars
                save_image_to_folder(image_path, segmented_chars_folder, filename, prefix, count, 0)
            
            elif is_latex and 0.50 < confidence < 0.95:
                new_char = segment_expression_to_images(image_path, 1, 1, 5, 120)
                # print (new_char)
                art_count += 1
                for i in new_char:
                    
                    is_latex1, confidence1, char1 = is_latex_match(i, latex_mapping, device)
                    saved_char.append(char1)
                    print(saved_char)
                
                for i in saved_char:
                    temp_arr.append(i)
                    
                    if len(temp_arr) == 3:
                        if check_pattern(temp_arr, keys):
                            temp_arr = separator.join(temp_arr)
                            final_arr.append(temp_arr)
                            temp_arr = []
                    elif len(temp_arr) > 3:
                        temp = temp_arr.pop(0)
                        final_arr.append(temp)
                
                for i in temp_arr:        
                    final_arr.append(i)
                
                print(final_arr)
                write_latex(final_arr, segmented_chars_folder, "char", count)
                
                          
            else:
                # Process the image with higher sensitivity segmentation
                segment_expression(image_path, segmented_chars_folder, 1, 1, 5, 120, count)
                
                
def extract_latex_from_segmented_images(segmented_chars_folder, latex_index_path, device):
    """
    Process all images in the `segmented_chars_folder` to extract LaTeX notation.
    Combines all recognized LaTeX symbols into a single expression.
    If a `.txt` file is encountered, its content is added directly to the expression.

    :param segmented_chars_folder: Path to the folder containing segmented character images.
    :param latex_index_path: Path to the JSON file containing LaTeX index mappings.
    :param device: Device to run the model ('cpu' or 'cuda').
    :return: Complete LaTeX string.
    """
    # Load LaTeX token mapping
    with open(latex_index_path, 'r') as f:
        latex_mapping = json.load(f)

    latex_expression = ""  # Initialize LaTeX expression

    for filename in sorted(os.listdir(segmented_chars_folder)):  # Sort to process in logical order
        image_path = os.path.join(segmented_chars_folder, filename)

        if os.path.isfile(image_path):
            # Check if the file is a .txt file
            if filename.endswith(".txt"):
                try:
                    # Open the .txt file and read its content
                    with open(image_path, "r") as txt_file:
                        txt_content = txt_file.read().strip()
                        latex_expression += txt_content
                        print(f"Found .txt file: {filename}, content: {txt_content}")
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
            else:
                try:
                    # Load image for character recognition
                    image = Image.open(image_path)

                    # Recognize character
                    char, confidence = model_character_recognition_alter(image, latex_mapping, device)
                    
                    # Only append the character if confidence is high
                    if confidence > 0.5:
                        latex_expression += char
                        print(f"{filename} - Recognized: {char}, Confidence: {confidence:.2f}")
                    else:
                        print(f"Low confidence for {filename}, skipping.")

                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    return latex_expression



def clean_latex_expression(raw_latex):
    """
    Clean and format a raw LaTeX expression for readability and correctness.
    :param raw_latex: Raw LaTeX string output.
    :return: Cleaned and formatted LaTeX string.
    """
    # Handle unrecognized tokens (<UNK>)
    cleaned = raw_latex.replace("<UNK>", "")
    
    # Replace repeating minus signs with equal signs
    cleaned = re.sub(r"-{2,}", "=", cleaned)
    
    # Ensure `\frac` has proper braces
    cleaned = re.sub(r"\\frac", r"\\frac{}", cleaned)  # Add empty braces if missing
    
    # Fix subscript and superscript braces
    cleaned = cleaned.replace("_", "_{").replace("^", "^{")
    cleaned = re.sub(r"_{([^}]*)", r"_{\1}", cleaned)  # Ensure `_` braces are closed
    cleaned = re.sub(r"\^{([^}]*)", r"^{\1}", cleaned)  # Ensure `^` braces are closed

    # Match `\left` and `\right` pairs and make sure they are balanced
    cleaned = re.sub(r"\\left\(", r"\\left(", cleaned)
    cleaned = re.sub(r"\\right\)", r"\\right)", cleaned)
    
    # Fix any unbalanced braces
    open_braces = cleaned.count("{")
    close_braces = cleaned.count("}")
    if open_braces > close_braces:
        cleaned += "}" * (open_braces - close_braces)
    elif close_braces > open_braces:
        cleaned = "{" * (close_braces - open_braces) + cleaned

    # Remove unnecessary characters like stray zeros or non-mathematical symbols
    cleaned = re.sub(r"[^\w\{\}\^\\\_\+\-\=\(\)\.\,]", "", cleaned)
    
    # Clean up the expression further, removing any extra or misplaced symbols
    cleaned = re.sub(r"\s+", " ", cleaned).strip()  # Remove excessive spaces
    
    # Wrap the expression in LaTeX math mode for proper display
    cleaned = r"[" + cleaned + r"]"

    return cleaned

def process_raw_latex(latex_string):
    """
    Process a raw LaTeX string to ensure it is in a valid format.
    :param latex_string: The raw LaTeX string.
    :return: Processed LaTeX string in valid format.
    """
    latex_string = clean_latex_expression(latex_string)
    
    # Handle any specific cases or corrections
    latex_string = latex_string.replace("lim_", "lim_{}")  # Fix for \lim cases if necessary
    latex_string = latex_string.replace("to", r"\to")  # Ensure \to is correctly formatted
    
    return latex_string

def model_process(image_path):
    
    # First Process
    # Path to your mathematical expression image
    segmented_chars_folder = 'segmented_chars'
    segmented_groups_folder = 'segmented_groups'
    latex_index_path = 'latex.json'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    clear_folder(segmented_groups_folder)
    clear_folder(segmented_chars_folder)
    clear_folder("artificial_chars")
    

    # Segment the image and get bounding boxes
    regions = segment_expression(image_path, segmented_groups_folder, 7, 3, 10, 150)

    # Load the image again to visualize the segmented characters
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
    else:
        grayscale = preprocess_color_image(image)  # Ensure it's grayscale for visualization

        # Show all segmented characters in a grid
        show_segmented_characters(grayscale, regions)

    # Second Process
    process_images_in_group(segmented_chars_folder, segmented_groups_folder, latex_index_path, device)

    # Extract LaTeX
    latex_result = extract_latex_from_segmented_images(segmented_chars_folder, latex_index_path, device)
    print("Final LaTeX Expression:")
    print(latex_result)

    formatted_latex = clean_latex_expression(latex_result)
    print(formatted_latex)
    
    return formatted_latex

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        # Check and handle the uploaded file
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the file
        image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(image_path)

        

        # Process the image using pytesseract
        extracted_text = model_process(image_path)
        
        return jsonify({'text': extracted_text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)