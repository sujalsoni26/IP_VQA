import torch
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
import json

# Load questions from JSON
with open("../../../NuScenes_val_questions.json", "r") as file:
    data = json.load(file)

# Filter relevant questions
questions_list = [
    obj["question"]
    for obj in data["questions"]
    if obj["sample_token"] == "2878a9ab393f42a2bbb426d8a14690d9" and obj["template_type"] == "count"
]

# Image paths with context labels
image_paths = [
    ("Front View", "../../../val_count_questions_images/2878a9ab393f42a2bbb426d8a14690d9_samples_CAM_FRONT_n008-2018-08-28-16-43-51-0400__CAM_FRONT__1535489341512404.jpg"),
    ("Front Left View", "../../../val_count_questions_images/2878a9ab393f42a2bbb426d8a14690d9_samples_CAM_FRONT_LEFT_n008-2018-08-28-16-43-51-0400__CAM_FRONT_LEFT__1535489341504799.jpg"),
    ("Front Right View", "../../../val_count_questions_images/2878a9ab393f42a2bbb426d8a14690d9_samples_CAM_FRONT_RIGHT_n008-2018-08-28-16-43-51-0400__CAM_FRONT_RIGHT__1535489341520482.jpg"),
    ("Back View", "../../../val_count_questions_images/2878a9ab393f42a2bbb426d8a14690d9_samples_CAM_BACK_n008-2018-08-28-16-43-51-0400__CAM_BACK__1535489341537558.jpg"),
    ("Back Right View", "../../../val_count_questions_images/2878a9ab393f42a2bbb426d8a14690d9_samples_CAM_BACK_RIGHT_n008-2018-08-28-16-43-51-0400__CAM_BACK_RIGHT__1535489341528113.jpg"),
    ("Back Left View", "../../../val_count_questions_images/2878a9ab393f42a2bbb426d8a14690d9_samples_CAM_BACK_LEFT_n008-2018-08-28-16-43-51-0400__CAM_BACK_LEFT__1535489341547405.jpg"),
]

# Initialize DeepSeek-VL model
model_path = "deepseek-ai/deepseek-vl-7b-chat"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# Prepare conversation history
conversation = []

# Load and process all images first
image_references = []
pil_images = []

for context, image_path in image_paths:
    print(f"Processing image: {context}")

    conversation.append({
        "role": "User",
        "content": f"Here is an image from the {context} of a car.",
        "images": [image_path]  # Attaching the image
    })
    
    image_references.append(image_path)

# Load all images
pil_images = load_pil_images(conversation)

# Acknowledge image processing before asking questions
conversation.append({
    "role": "Assistant",
    "content": "I have analyzed all six images. You can now ask me questions based on them."
})

# Now, ask questions
for question in questions_list:
    print("Question:", question)

    # Append user question
    conversation_with_question = conversation + [
        {"role": "User", "content": f"Based on all six images, give single integer and no text as response for, {question}"},
        {"role": "Assistant", "content": ""}
    ]

    # Prepare inputs
    prepare_inputs = vl_chat_processor(
        conversations=conversation_with_question,
        images=pil_images,
        force_batchify=True
    ).to(vl_gpt.device)

    # Run model inference
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True
    )

    # Decode response
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    print("Response:", answer, "\n")

