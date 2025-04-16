




import os
import cv2
import torch
import zipfile
from flask import Flask, render_template, request, send_from_directory
from transformers import BlipForConditionalGeneration, AutoProcessor
from huggingface_hub import InferenceClient
from PIL import Image
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Securely load API key from environment
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# 1. Extract image information
def extract_image_info(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return {
        "number_of_contours": len(contours),
        "image_size": image.shape
    }

# 2. Generate image
def generate_image(text_prompt, image_info=None):
    combined_prompt = text_prompt
    if image_info:
        combined_prompt += f". Include {image_info['number_of_contours']} objects."

    client = InferenceClient(api_key=HUGGINGFACE_API_KEY)
    image = client.text_to_image(combined_prompt, model="black-forest-labs/FLUX.1-schnell")

    image_path = os.path.join(UPLOAD_FOLDER, "generated_image.jpg")
    image.save(image_path)
    return image_path

# 3. Caption generation
def generate_caption(image_path, user_prompt):
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    image = Image.open(image_path)
    prompt = f"Create a compelling advertisement caption for this image based on: {user_prompt}."
    inputs = processor(image, prompt, return_tensors="pt")
    output = model.generate(**inputs)

    return processor.decode(output[0], skip_special_tokens=True)

# 4. Enhance caption
def enhance_caption(caption, user_prompt):
    client = InferenceClient(api_key=HUGGINGFACE_API_KEY)
    prompt = f"Improve this caption to be persuasive and emotional for a product about '{user_prompt}': {caption}"

    messages = [{"role": "user", "content": prompt}]
    completion = client.chat_completion(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=messages,
        max_tokens=100
    )

    return completion.choices[0]["message"]["content"].strip()

# 5. Save caption to file
def save_caption_to_file(caption):
    path = os.path.join(UPLOAD_FOLDER, "generated_caption.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(caption)
    return path

# 6. Create ZIP file for storyboard download
def create_storyboard_zip(image_path, caption_path):
    zip_path = os.path.join(UPLOAD_FOLDER, "storyboard.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.write(image_path, os.path.basename(image_path))
        zipf.write(caption_path, os.path.basename(caption_path))
    return zip_path

# --- Main route ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_text = request.form["prompt"]
        action = request.form.get("action")
        feedback = request.form.get("feedback")
        image_file = request.files.get("image")

        image_info = None

        if action == "generate_with_image" and image_file:
            uploaded_path = os.path.join(UPLOAD_FOLDER, "uploaded_image.jpg")
            image_file.save(uploaded_path)
            image_info = extract_image_info(uploaded_path)

        final_prompt = f"{user_text}. Changes requested: {feedback}" if feedback else user_text

        # Generate image and caption
        ai_image_path = generate_image(final_prompt, image_info)
        raw_caption = generate_caption(ai_image_path, final_prompt)
        improved_caption = enhance_caption(raw_caption, final_prompt)

        caption_path = save_caption_to_file(improved_caption)
        zip_path = create_storyboard_zip(ai_image_path, caption_path)

        return render_template(
            "index.html",
            image=ai_image_path,
            caption=improved_caption,
            zip_file=os.path.basename(zip_path),
            original_prompt=user_text
        )

    return render_template("index.html", image=None, caption=None, zip_file=None, original_prompt=None)

# Download route for the storyboard zip
@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

# if __name__ == "__main__":
#     app.run(debug=True)
