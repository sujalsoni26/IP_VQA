import json
import os
from openai import OpenAI
import base64
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("API_KEY")

if not token:
    raise ValueError("Errrorrr reading API key")

endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o"

def get_image_data_url(image_file: str, image_format: str) -> str:
    """Converts an image file to a base64 data URL."""
    try:
        with open(image_file, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/{image_format};base64,{image_data}"
    except FileNotFoundError:
        print(f"Could not read '{image_file}'.")
        return None

# Define image paths with context descriptions
# sample_token = "0d45f0bedc6d455ea5a28cb4939c910d"
sample_token = "2878a9ab393f42a2bbb426d8a14690d9"
# image_path= {
#    "front": "../../val_count_questions_images/0d45f0bedc6d455ea5a28cb4939c910d_samples_CAM_FRONT_n015-2018-10-08-15-44-23+0800__CAM_FRONT__1538984917912460.jpg",
#    "front_left": "../../val_count_questions_images/0d45f0bedc6d455ea5a28cb4939c910d_samples_CAM_FRONT_LEFT_n015-2018-10-08-15-44-23+0800__CAM_FRONT_LEFT__1538984917904844.jpg",
#    "front_right": "../../val_count_questions_images/0d45f0bedc6d455ea5a28cb4939c910d_samples_CAM_FRONT_RIGHT_n015-2018-10-08-15-44-23+0800__CAM_FRONT_RIGHT__1538984917920339.jpg",
#    "back": "../../val_count_questions_images/0d45f0bedc6d455ea5a28cb4939c910d_samples_CAM_BACK_n015-2018-10-08-15-44-23+0800__CAM_BACK__1538984917937525.jpg",
#    "back_left": "../../val_count_questions_images/0d45f0bedc6d455ea5a28cb4939c910d_samples_CAM_BACK_LEFT_n015-2018-10-08-15-44-23+0800__CAM_BACK_LEFT__1538984917947423.jpg",
#    "back_right": "../../val_count_questions_images/0d45f0bedc6d455ea5a28cb4939c910d_samples_CAM_BACK_RIGHT_n015-2018-10-08-15-44-23+0800__CAM_BACK_RIGHT__1538984917927893.jpg",
# }
image_path= {
    "front": "../../val_count_questions_images/2878a9ab393f42a2bbb426d8a14690d9_samples_CAM_FRONT_n008-2018-08-28-16-43-51-0400__CAM_FRONT__1535489341512404.jpg",
    "front_left": "../../val_count_questions_images/2878a9ab393f42a2bbb426d8a14690d9_samples_CAM_FRONT_LEFT_n008-2018-08-28-16-43-51-0400__CAM_FRONT_LEFT__1535489341504799.jpg",
    "front_right": "../../val_count_questions_images/2878a9ab393f42a2bbb426d8a14690d9_samples_CAM_FRONT_RIGHT_n008-2018-08-28-16-43-51-0400__CAM_FRONT_RIGHT__1535489341520482.jpg",
    "back": "../../val_count_questions_images/2878a9ab393f42a2bbb426d8a14690d9_samples_CAM_BACK_n008-2018-08-28-16-43-51-0400__CAM_BACK__1535489341537558.jpg",
    "back_left": "../../val_count_questions_images/2878a9ab393f42a2bbb426d8a14690d9_samples_CAM_BACK_LEFT_n008-2018-08-28-16-43-51-0400__CAM_BACK_LEFT__1535489341547405.jpg",
    "back_right": "../../val_count_questions_images/2878a9ab393f42a2bbb426d8a14690d9_samples_CAM_BACK_RIGHT_n008-2018-08-28-16-43-51-0400__CAM_BACK_RIGHT__1535489341528113.jpg",
}
image_paths = {
    "front": {
        "path": image_path["front"],
        "context": "This image is from the front camera of a vehicle."
    },
    "front_left": {
        "path": image_path["front_left"],
        "context": "This image is from the front-left camera of a vehicle."
    },
    "front_right": {
        "path": image_path["front_right"],
        "context": "This image is from the front-right camera of a vehicle."
    },
    "back": {
        "path": image_path["back"],
        "context": "This image is from the back camera of a vehicle."
    },
    "back_left": {
        "path": image_path["back_left"],
        "context": "This image is from the back-left camera of a vehicle."
    },
    "back_right": {
        "path": image_path["back_right"],
        "context": "This image is from the back-right camera of a vehicle."
    }
}

image_data_urls = {
    key: {
        "url": get_image_data_url(info["path"], "jpg"),
        "context": info["context"]
    }
    for key, info in image_paths.items()
}


json_file = "../../NuScenes_val_questions.json"
questions_list = []
if os.path.exists(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)
        questions_list = [
            obj["question"]
            for obj in data.get("questions", [])
            if obj.get("sample_token") == sample_token
            and obj.get("template_type") == "count"
        ]
else:
    print(f"Warning: JSON file '{json_file}' not found.")

messages = [
    {"role": "system", "content": "You are a helpful assistant that answers count-related questions based on images."},
    {"role": "user", "content": [{"type": "text", "text": "Write question and then answer in a single numeric value - " +  q} for q in questions_list]}
#   {"role": "user", "content": [{"type": "text", "text": "Give detailed analysis of the images and then answer the question -  " +  q} for q in questions_list]}
]

for img_data in image_data_urls.values():
    messages[1]["content"].append(
        {"type": "text", "text": img_data["context"]}
    )
    messages[1]["content"].append(
        {"type": "image_url", "image_url": {"url": img_data["url"], "detail": "high"}}
    )

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

response = client.chat.completions.create(
    messages=messages,
    model=model_name,
)

print(response.choices[0].message.content)
