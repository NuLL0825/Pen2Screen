import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

import system
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch
from PIL import Image
import json
import torch.nn.functional as F
import shutil

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

    # Visualize the original image
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
    
    detected_symbols = detect_symbols(image, latex_mapping, device)  # Function to detect symbols in the image
    for symbol in latex_key.keys():
        if symbol == detected_symbols and symbol in keys:
            return True
    return False

def detect_symbols(image, latex_mapping, device):
    """
    Placeholder function to detect symbols in an image.
    This needs to be implemented based on your model or processing pipeline.
    """
    # This is where you would integrate your text recognition or symbol detection logic.
    # For now, we assume a list of detected symbols for the example.
    char, confidence = model_character_recognition_alter(image, latex_mapping, device)
    return char  # Example detected symbols

def process_images_in_group(segmented_chars_folder, segmented_groups_folder, latex_index_path, device): # second process
    """
    Process each image in the segmented_groups folder, check for LaTeX matches,
    move matched images to segmented_chars, and then perform character segmentation
    with higher sensitivity.
    """
    with open(latex_index_path, 'r') as f:
        latex_mapping = json.load(f)
    count = 0
    prefix = "char"
    
    for filename in os.listdir(segmented_groups_folder):
        image_path = os.path.join(segmented_groups_folder, filename)
        
        
        if os.path.isfile(image_path):
            # image = cv2.imread(image_path)
            image = Image.open(image_path)
            count += 1
            
            
            # Check if the image contains any LaTeX symbols from latex_mapping
            if is_latex_match(image, latex_mapping, device):
                # Move the image to segmented_chars
                save_image_to_folder(image_path, segmented_chars_folder, filename, prefix, count, 0)
                
                          
            else:
                # Process the image with higher sensitivity segmentation
                segment_expression(image_path, segmented_chars_folder, 1, 1, 5, 120, count)

def extract_latex_from_segmented_images(segmented_chars_folder, latex_index_path, device):
    """
    Process all images in the `segmented_chars_folder` to extract LaTeX notation.
    Combines all recognized LaTeX symbols into a single expression.
    
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
            try:
                # Load image
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
    Clean and format a raw LaTeX expression for readability.
    :param raw_latex: Raw LaTeX string output.
    :return: Cleaned and formatted LaTeX string.
    """
    # Handle unrecognized tokens (<UNK>)
    cleaned = raw_latex.replace("<UNK>", r"")  # Replace with multiplication (or adjust as needed)
    
    # Format subscripts and superscripts
    cleaned = cleaned.replace("_", "_{").replace("^", "^{")
    cleaned = cleaned.replace("{ ", "{").replace(" }", "}")  # Fix any spacing issues
    
    # Ensure subscripts/superscripts are closed properly
    stack = []
    for i, char in enumerate(cleaned):
        if char in "_^" and (i + 1 < len(cleaned) and cleaned[i + 1] != "{"):
            stack.append(i)
        if char == "}" and stack:
            stack.pop()

    for i in reversed(stack):
        cleaned = cleaned[:i + 1] + "{" + cleaned[i + 1] + "}" + cleaned[i + 2:]

    # Wrap in LaTeX math mode
    cleaned = r"[" + cleaned + r"]"
    
    return cleaned

def model_process(image_path):
    
    # First Process
    # Path to your mathematical expression image
    segmented_chars_folder = 'segmented_chars'
    segmented_groups_folder = 'segmented_groups'
    latex_index_path = 'latex.json'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    clear_folder(segmented_groups_folder)
    clear_folder(segmented_chars_folder)

    # Segment the image and get bounding boxes
    regions = segment_expression(image_path, segmented_groups_folder, 6, 5, 10, 70)

    # Load the image again to visualize the segmented characters
    image = cv2.imread(image_path)

    # Second Process
    process_images_in_group(segmented_chars_folder, segmented_groups_folder, latex_index_path, device)

    # Paths and device setup
    segmented_chars_folder = "segmented_chars"
    latex_index_path = "latex.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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