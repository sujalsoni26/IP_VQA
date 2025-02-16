import torch
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
import json

# Load questions from JSON
with open("../../../NuScenes_val_questions.json", "r") as file:
    data = json.load(file)

# Filter questions
questions_list = [
    obj["question"]
    for obj in data["questions"]
    if obj["sample_token"] == "0d45f0bedc6d455ea5a28cb4939c910d" and obj["template_type"] == "count"
]

# Image paths (6 images)
image_paths = [
    "../../../val_count_questions_images/0d45f0bedc6d455ea5a28cb4939c910d_samples_CAM_FRONT_n015-2018-10-08-15-44-23+0800__CAM_FRONT__1538984917912460.jpg",
    "../../../val_count_questions_images/0d45f0bedc6d455ea5a28cb4939c910d_samples_CAM_FRONT_LEFT_n015-2018-10-08-15-44-23+0800__CAM_FRONT_LEFT__1538984917904844.jpg",
    "../../../val_count_questions_images/0d45f0bedc6d455ea5a28cb4939c910d_samples_CAM_FRONT_RIGHT_n015-2018-10-08-15-44-23+0800__CAM_FRONT_RIGHT__1538984917920339.jpg",
    "../../../val_count_questions_images/0d45f0bedc6d455ea5a28cb4939c910d_samples_CAM_BACK_n015-2018-10-08-15-44-23+0800__CAM_BACK__1538984917937525.jpg",
    "../../../val_count_questions_images/0d45f0bedc6d455ea5a28cb4939c910d_samples_CAM_BACK_RIGHT_n015-2018-10-08-15-44-23+0800__CAM_BACK_RIGHT__1538984917927893.jpg",
    "../../../val_count_questions_images/0d45f0bedc6d455ea5a28cb4939c910d_samples_CAM_BACK_LEFT_n015-2018-10-08-15-44-23+0800__CAM_BACK_LEFT__1538984917947423.jpg"
]

# Initialize DeepSeek-VL model
model_path = "deepseek-ai/deepseek-vl-7b-chat"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# Iterate over each image separately
for image_path in image_paths:
    print("\nProcessing image:", image_path)

    for question in questions_list:
        print("Question:", question)

        # Create conversation with one image at a time
        conversation = [
            {
                "role": "User",
                "content": "<image_placeholder> " + question,
                "images": [image_path]  # Single image
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]

        # Load single image
        pil_images = load_pil_images(conversation)

        # Prepare inputs
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(vl_gpt.device)

        # Run model
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # Generate response
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

