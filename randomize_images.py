import os
import cv2
import random
import uuid

# Define customizable variables
input_dir = os.path.join(os.getcwd(), "raw")  # Input directory
output_dir = os.path.join(os.getcwd(), "processed")  # Output directory
num_variations = 50  # Number of variations to create for each image

# Random transformation limits
rotation_limit = 180  # Maximum rotation angle in degrees
resize_min = 0.5  # Minimum scale factor for resizing
resize_max = 2.0  # Maximum scale factor for resizing
brightness_min = 0.5  # Minimum brightness factor
brightness_max = 1.5  # Maximum brightness factor
contrast_min = -25  # Minimum contrast factor
contrast_max = 25  # Maximum contrast factor
saturation_min = 0.5  # Minimum saturation factor
saturation_max = 1.5  # Maximum saturation factor
exposure_min = 0.5  # Minimum exposure factor
exposure_max = 1.5  # Maximum exposure factor

# Define the output name suffix
output_name_suffix = "variation"

# Function to apply random transformations to an image
def randomize_image(image):
    # Randomly rotate the image (-rotation_limit to +rotation_limit degrees)
    angle = random.randint(-rotation_limit, rotation_limit)
    M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Randomly flip the image horizontally
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    # Randomly resize the image (resize_min to resize_max of original size)
    scale_factor = random.uniform(resize_min, resize_max)
    image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

    # Randomly apply brightness, contrast, saturation, and exposure adjustments
    alpha = random.uniform(brightness_min, brightness_max)  # Brightness
    beta = random.uniform(contrast_min, contrast_max)      # Contrast

    # Saturation and exposure adjustments
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = hsv_image[:, :, 1] * random.uniform(saturation_min, saturation_max)  # Saturation
    hsv_image[:, :, 2] = hsv_image[:, :, 2] * random.uniform(exposure_min, exposure_max)    # Exposure

    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return image

if not os.path.exists(input_dir):
    os.makedirs(input_dir)

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process all images in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith((".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG")):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        print(f"Processing {input_path}...")

        # Load the image
        image = cv2.imread(input_path)

        # Apply random transformations to create variations
        for i in range(num_variations):
            randomized_image = randomize_image(image)

            # Generate a random UUID
            random_uuid = str(uuid.uuid4())

            # Define the variation output name
            variation_output_name = f"{os.path.splitext(filename)[0]}_{output_name_suffix}_{random_uuid}.jpg"

            # Save the randomized image with the variation output name
            variation_output_path = os.path.join(output_dir, variation_output_name)
            cv2.imwrite(variation_output_path, randomized_image)

            print(f"Variation {i + 1}")
