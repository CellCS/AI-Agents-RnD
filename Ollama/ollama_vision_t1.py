import ollama
import pprint
import json
import base64
import requests
from PIL import Image

# ollama run llama3.2-vision
# use 'ollama list' get model name
# https://github.com/ollama/ollama/blob/main/docs/api.md
VISION_MODEL = 'llama3.2-vision'
OLLAMA_CHAT_URLS=["http://localhost:11434/api/chat", "http://localhost:11434/api/generate",]
OLLAMA_CHAT_URL="http://localhost:11434/api/chat"

SYSTEM_PROMPT = """Act as an OCR assistant. Analyze the provided image and:
1. Recognize all visible text in the image as accurately as possible.
2. Maintain the original structure and formatting of the text.
3. If any words or phrases are unclear, indicate this with [unclear] in your transcription.
Provide only the transcription without any additional comments."""

def encode_image_to_base64(image_path):
    """Convert an image file to a base64 encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def perform_ocr_imagepath(image_path, systemprompt, usingdefault:bool):
    """OCR one image."""
    base64_image = encode_image_to_base64(image_path)
    if usingdefault is True:
        return perform_ocr_default(base64_image, systemprompt)
    return perform_ocr(base64_image, systemprompt)
    
def perform_ocr(base64_image, systemprompt):
    """Using request with api url."""
    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": VISION_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": systemprompt,
                        "images": [base64_image],
                    },
                ],
                "stream":False
            }
        )
        if response.status_code == 200:
            pprint.pprint(response.content)
        else:
            pprint.pprint("Error:", response.content)
        return response
    except Exception as e:
        return e
    
def perform_ocr_default(base64_image, systemprompt):
    """Using ollama lib to call."""
    response = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {
                'role': 'user',
                'content':systemprompt,
                'images': [base64_image]
            }
        ]
    )
    return response
    
if __name__ == "__main__":
    image_path = "./assets/oneform.png"
    context_jsonfile='./assets/formsetup.json'
    # get plate setup basic info
    with open(context_jsonfile) as f:
        formsetupstring = json.load(f)
    systemprompt = f'''Given a CRF form image and a corresponding form setup JSON content with field setup information. Here is setup info: {formsetupstring}. Please analyze the image to identify each field and its corresponding value. For checkboxes or choice fields, the setup JSON file provides the available choices, with the value indicating whether each choice is selected or unselected. Please return a JSON object with the filled field values from the provided form image.'''
    response = perform_ocr_imagepath(image_path, systemprompt, False)
    if response:
        pprint.pprint(response)
    

