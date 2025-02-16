import google.generativeai as genai
import json
import base64
from dotenv import load_dotenv
import os

# Load JSON file
with open("../../NuScenes_val_questions.json", "r") as file:
    data = json.load(file)

# Extract relevant questions
# sample_token = "0d45f0bedc6d455ea5a28cb4939c910d"
sample_token1 = "2878a9ab393f42a2bbb426d8a14690d9"
questions_list = [
    obj["question"]
    for obj in data["questions"]
    if obj["sample_token"] == sample_token1 and obj["template_type"] == "count"
]

# Image paths
# image_paths = [
#     "../../val_count_questions_images/0d45f0bedc6d455ea5a28cb4939c910d_samples_CAM_FRONT_n015-2018-10-08-15-44-23+0800__CAM_FRONT__1538984917912460.jpg",
#     "../../val_count_questions_images/0d45f0bedc6d455ea5a28cb4939c910d_samples_CAM_FRONT_LEFT_n015-2018-10-08-15-44-23+0800__CAM_FRONT_LEFT__1538984917904844.jpg",
#     "../../val_count_questions_images/0d45f0bedc6d455ea5a28cb4939c910d_samples_CAM_FRONT_RIGHT_n015-2018-10-08-15-44-23+0800__CAM_FRONT_RIGHT__1538984917920339.jpg",
#     "../../val_count_questions_images/0d45f0bedc6d455ea5a28cb4939c910d_samples_CAM_BACK_n015-2018-10-08-15-44-23+0800__CAM_BACK__1538984917937525.jpg",
#     "../../val_count_questions_images/0d45f0bedc6d455ea5a28cb4939c910d_samples_CAM_BACK_LEFT_n015-2018-10-08-15-44-23+0800__CAM_BACK_LEFT__1538984917947423.jpg",
#     "../../val_count_questions_images/0d45f0bedc6d455ea5a28cb4939c910d_samples_CAM_BACK_RIGHT_n015-2018-10-08-15-44-23+0800__CAM_BACK_RIGHT__1538984917927893.jpg"
# ]
image_paths = [
    "../../val_count_questions_images/2878a9ab393f42a2bbb426d8a14690d9_samples_CAM_FRONT_n008-2018-08-28-16-43-51-0400__CAM_FRONT__1535489341512404.jpg",
    "../../val_count_questions_images/2878a9ab393f42a2bbb426d8a14690d9_samples_CAM_FRONT_LEFT_n008-2018-08-28-16-43-51-0400__CAM_FRONT_LEFT__1535489341504799.jpg",
    "../../val_count_questions_images/2878a9ab393f42a2bbb426d8a14690d9_samples_CAM_FRONT_RIGHT_n008-2018-08-28-16-43-51-0400__CAM_FRONT_RIGHT__1535489341520482.jpg",
    "../../val_count_questions_images/2878a9ab393f42a2bbb426d8a14690d9_samples_CAM_BACK_n008-2018-08-28-16-43-51-0400__CAM_BACK__1535489341537558.jpg",
    "../../val_count_questions_images/2878a9ab393f42a2bbb426d8a14690d9_samples_CAM_BACK_LEFT_n008-2018-08-28-16-43-51-0400__CAM_BACK_LEFT__1535489341547405.jpg",
    "../../val_count_questions_images/2878a9ab393f42a2bbb426d8a14690d9_samples_CAM_BACK_RIGHT_n008-2018-08-28-16-43-51-0400__CAM_BACK_RIGHT__1535489341528113.jpg"
]

# Hardcoded context
context_text = """
1st image is the front camera image taken from the vehicle.
2nd image is the front-left camera image taken from the vehicle.
3rd image is the front-right camera image taken from the vehicle.
4th image is the rear camera image taken from the vehicle.
5th image is the rear-left camera image taken from the vehicle.
6th image is the rear-right camera image taken from the vehicle.
"""

# Function to encode images in base64
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Prepare image data for Gemini model
image_data = [{"mime_type": "image/jpeg", "data": encode_image(img_path)} for img_path in image_paths]

# Set up Gemini API key
# Load .env file
load_dotenv()
# Get API key
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)

# Initialize Gemini model
model = genai.GenerativeModel("gemini-2.0-flash")

# Construct a single input with all images, context, and questions
input_content = [
    {"text": f"Context for images:\n{context_text}\n\nAnswer the following questions based on the images with a single numeric value:"}
] + image_data + [{"text": f"\n".join(questions_list)}]

# Generate response from Gemini
response = model.generate_content(input_content)

# Print response
print("Response:", response.text)
