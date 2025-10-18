from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import requests 
import torch

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch
import requests

# Model, processor ve tokenizer'ı yükle
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")


img_url = "https://www.refinery29.com/images/9760915.jpg"
image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")


pixel_values = processor(images=image, return_tensors="pt").pixel_values


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
pixel_values = pixel_values.to(device)


output_ids = model.generate(pixel_values, max_length=32, num_beams=4)
caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("ViT-GPT2 Açıklaması:", caption)
