from PIL import Image
import os

def binarize_image(input_image_path, output_image_path, threshold=128):
    try:
        # Check if the input file exists
        if not os.path.isfile(input_image_path):
            print(f"Error: The file {input_image_path} does not exist.")
            return
        
        # Open the image file
        with Image.open(input_image_path) as img:
            # Convert the image to grayscale
            gray_img = img.convert("L")
            
            # Binarize the image using the specified threshold
            binarized_img = gray_img.point(lambda x: 255 if x > threshold else 0, '1')
            
            # Save the binarized image
            binarized_img.save(output_image_path)
            print(f"Binarized image saved as: {output_image_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage (for testing)
if __name__ == '__main__':
    input_path = 'D:/Maui/Courses - 4th YR 1st sem/Thesis 2/MobileApplication/flask_app/image_sample/burgir.jpg'  # Change to your input image path
    output_path = 'D:/Maui/Courses - 4th YR 1st sem/Thesis 2/MobileApplication/flask_app/image_sample/binarized_image.png'  # Change to your desired output path
    binarize_image(input_path, output_path)