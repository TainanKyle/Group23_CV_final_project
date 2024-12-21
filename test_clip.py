import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from diffusers import ControlNetModel
from diffusers import StableDiffusionControlNetPipeline as DiffusionPipeline
device = "cuda:1" if torch.cuda.is_available() else "cpu"

controlnet_name = "lllyasviel/control_v11f1p_sd15_depth"

controlnet = ControlNetModel.from_pretrained(controlnet_name)
checkpoint_name = "runwayml/stable-diffusion-v1-5"
diffusion_model = DiffusionPipeline.from_pretrained(checkpoint_name, controlnet=controlnet).to(device)
text = "a Japanese style bedroom"
text_input = diffusion_model.tokenizer(text, padding="max_length", max_length=diffusion_model.tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
text_embeds = diffusion_model.text_encoder(text_input)[0]
text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
text_embeds = text_embeds.reshape(text_embeds.shape[1], text_embeds.shape[2])

# 載入預訓練的 CLIP 模型和處理器
# model_name = "openai/clip-vit-base-patch32"
model_name = "openai/clip-vit-large-patch14"
# model_name = "openai/clip-vit-large-patch14-336"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# 檢查是否有 GPU 可用
model = model.to(device)

# 讀取圖片
image_path = "/home/ado/storage/SceneTex/a_Bohemian_style_bedroom.jpg"  # 替換成你的圖片路徑
image = Image.open(image_path).convert("RGB")

# 定義文字描述
# text = ["a Japanese style living room", "a Bohemian style living room", "a country style living room"]
# text = ["a Japanese style bedroom", "a Bohemian style bedroom", "a country style bedroom", "a Scandinavian style bedroom"]
# text = ["Japanese style", "Bohemian style", "country style", "Scandinavian style"]
# 使用處理器對圖片和文字進行預處理
inputs = processor(text=text, max_length=77, return_tensors="pt", padding="max_length", truncation=True).to(device)

# 使用模型進行特徵提取
with torch.no_grad():
    outputs = model(**inputs)
    # image_features = outputs.image_embeds  # 圖像特徵
    text_features = outputs.text_embeds    # 文字特徵

# 正規化特徵向量
# image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
text_features = text_features.repeat(1, text_embeds.shape[0], 1)
text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
text_features = text_features.reshape(text_features.shape[1], text_features.shape[2])

print(text_features.shape)
print(text_embeds.shape)
# 計算餘弦相似度
similarity = torch.matmul(text_features, text_embeds.T)  # 相似度矩陣
print(similarity)
euclidean_distances = torch.norm(text_features - text_embeds, dim=-1)
print(euclidean_distances)

# import torch
# import clip
# from PIL import Image

# device = "cuda:1" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# image = preprocess(Image.open("a_country_style_living_room.jpg")).unsqueeze(0).to(device)
# text = clip.tokenize(["a Japanese style living room", "a Bohemian style living room", "a country style living room"]).to(device)

# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
    
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()


# print("Label logits:", logits_per_image)
# print("Label probs:", probs)