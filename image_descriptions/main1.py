from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests 
import torch 

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
img_url1 = "https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjk8p-Us8f5SXF4xKay8SFiQKvQyrzqYrOicQaPQiYimDazZvulAyjrK1c9G0WRLEZrdbD7dXNSTgYFYvb3vJA3hfQhVIiFGxu-nZ4lrNhABrHzx6-QtJIS3710dfFnpYqytM5mhrkhacU7/s320/7.jpg"
image = Image.open(requests.get(img_url1, stream = True).raw).convert("RGB")

inputs = processor(image, return_tensors = "pt")

with torch.no_grad():
    output = model.generate(**inputs)

caption = processor.decode(output[0], skip_special_tokens = True)
print(f"Üretilen Metin: {caption}") 