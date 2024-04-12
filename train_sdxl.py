import torch
from torch.utils.data import DataLoader
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import StableDiffusionPipeline
from torchvision import transforms
import os
from tqdm import tqdm
from PIL import Image
from model import TSC

# Define the dataset
class ImageCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, tokenizer, max_length=128):
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.filenames = [f for f in os.listdir(dataset_path) if f.endswith('.txt')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        caption_file = self.filenames[idx]
        caption_path = os.path.join(self.dataset_path, caption_file)
        
        with open(caption_path, 'r', encoding='utf-8') as file:
            caption = file.read().strip()

        image_name = caption_file.rsplit('.', 1)[0]
        image_path = os.path.join(self.dataset_path, f"{image_name}.png")
        
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        text_inputs = self.tokenizer(caption, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        text_input_ids = text_inputs.input_ids.squeeze()
        attention_mask = text_inputs.attention_mask.squeeze()
        
        return image, text_input_ids, attention_mask

def train_ella(dataset_path, tokenizer, llm_model, sdxl_path, save_path, epochs=10, batch_size=4, lr=1e-5):
    dataset = ImageCaptionDataset(dataset_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    sdxl_model = StableDiffusionPipeline.from_pretrained(sdxl_path, torch_dtype=torch.float16).to("cuda")
    sdxl_model.safety_checker = None
    sdxl_model.enable_xformers_memory_efficient_attention()
    
    tsc = TSC(llm_model.config).to("cuda")
    
    optimizer = torch.optim.AdamW(tsc.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader)*epochs)
    
    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, input_ids, attention_mask in progress_bar:
            images = images.to("cuda")
            input_ids = input_ids.to("cuda")
            attention_mask = attention_mask.to("cuda")
            
            with torch.no_grad():
                llm_outputs = llm_model(input_ids=input_ids, attention_mask=attention_mask)
                text_embeddings = llm_outputs.last_hidden_state
            
            timesteps = torch.randint(0, sdxl_model.scheduler.config.num_train_timesteps, (images.shape[0],), device=images.device).long()
            
            with torch.no_grad():
                latents = sdxl_model.vae.encode(images.half()).latent_dist.sample()
                latents = latents * 0.18215
                noise = torch.randn_like(latents)
                noised_latents = sdxl_model.scheduler.add_noise(latents, noise, timesteps)
            
            encoder_hidden_states = tsc(text_embeddings, timesteps)
            
            noise_pred = sdxl_model.unet(noised_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({"Loss": loss.item()})

    torch.save(tsc.state_dict(), save_path)
     
if __name__ == "__main__":
    dataset_path = "path to dataset"
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    llm_model = T5EncoderModel.from_pretrained("google/flan-t5-xl").to("cuda")
    sdxl_path = "path to diffusers sdxl"
    save_path = "./tsc.pth"
    
    train_ella(dataset_path, tokenizer, llm_model, sdxl_path, save_path)
