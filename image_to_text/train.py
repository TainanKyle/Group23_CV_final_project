import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import glob
import os
from tqdm import tqdm
from diffusers import ControlNetModel
from diffusers import StableDiffusionControlNetPipeline as DiffusionPipeline

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
epochs = 50
learning_rate = 1e-4
batch_size = 32
is_Train = True
is_Test = True

image_dir = "/home/ado/storage/interior/img"
text_dir = "/home/ado/storage/interior/style"
checkpoint_dir = "/home/ado/storage/CV_final_project/image_to_text/checkpoint"
test_image_dir = "/home/ado/storage/surface_relabel/val/img"
test_text_dir = "/home/ado/storage/surface_relabel/val/style"

class ImageTextDataset(Dataset):
    def __init__(self, image_dir, text_dir):
        # self.image_dir = image_dir
        # self.text_dir = text_dir
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        self.text_paths = sorted(glob.glob(os.path.join(text_dir, "*.txt")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        text_path = self.text_paths[idx]

        # image = Image.open(image_path).convert("RGB")
        text = open(text_path, "r").readline().strip()
        text += " style"
        # return image, text
        return image_path, text

class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1024, output_dim=768, seq_len=77):
        super(TwoLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim * seq_len)
        self.activation = nn.ReLU()
        self.seq_len = seq_len
        self.output_dim = output_dim

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)  # (batch_size, 77 * 768)
        x = x.view(-1, self.seq_len, self.output_dim)  # (batch_size, 77, 768)
        return x


clip_model_name = "openai/clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

controlnet_name = "lllyasviel/control_v11f1p_sd15_depth"
controlnet = ControlNetModel.from_pretrained(controlnet_name)
checkpoint_name = "runwayml/stable-diffusion-v1-5"
diffusion_model = DiffusionPipeline.from_pretrained(checkpoint_name, controlnet=controlnet).to(device)

dataset = ImageTextDataset(image_dir, text_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = TwoLayerMLP().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train
if is_Train:
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        # for images, texts in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        for image_paths, texts in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            # get image embeddings
            images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
            inputs = clip_processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                image_embeddings = clip_model.get_image_features(**inputs)  # (batch_size, 768)
                text_inputs = diffusion_model.tokenizer(texts, padding="max_length", max_length=diffusion_model.tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
                text_embeddings = diffusion_model.text_encoder(text_inputs)[0]

            # forward pass
            image_embeddings = image_embeddings.to(device)
            outputs = model(image_embeddings)

            loss = criterion(outputs, text_embeddings)
            running_loss += loss.item()

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader)}")

        # save the model of the lowest loss
        if epoch == 0:
            lowest_loss = running_loss
        else:
            if running_loss < lowest_loss:
                lowest_loss = running_loss
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "checkpoint_lowest_loss.pth"))

    # save the model
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "checkpoint_epoch_{epochs}.pth"))

# test
if is_Test:
    print("=> testing...")
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "checkpoint_lowest_loss.pth")))
    model.eval()
    with torch.no_grad():
        test_dataset = ImageTextDataset(test_image_dir, test_text_dir)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        total_similarity = 0.0
        total_loss = 0.0
        total_ecl_distance = 0.0
        for image_paths, texts in test_dataloader:
            images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
            inputs = clip_processor(images=images, return_tensors="pt").to(device)
            image_embeddings = clip_model.get_image_features(**inputs)
            outputs = model(image_embeddings)
            text_inputs = diffusion_model.tokenizer(texts, padding="max_length", max_length=diffusion_model.tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
            text_embeddings = diffusion_model.text_encoder(text_inputs)[0]

            # calculate average cosine similarity
            loss = criterion(outputs, text_embeddings)
            total_loss += loss.item()
            similarity = torch.cosine_similarity(outputs, text_embeddings, dim=-1).mean()
            total_similarity += similarity.item() 
            ecl_distance = torch.norm(outputs - text_embeddings, dim=-1).mean()
            total_ecl_distance += ecl_distance.item()
        print(f"Average Loss: {total_loss/len(test_dataloader)}")
        print(f"Average Cosine Similarity: {total_similarity/len(test_dataloader)}")    
        print(f"Average Euclidean Distance: {total_ecl_distance/len(test_dataloader)}")