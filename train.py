import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import T5Tokenizer
from model import ELLA, T5TextEmbedder
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers.image_processor import VaeImageProcessor
import tqdm

class ImageCaptionDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        
        # Define the transform to convert PIL image to tensor
        self.transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to a fixed size, e.g., (224, 224)
        transforms.ToTensor()
    ])

        # Assuming captions and images are in the same directory and have the same name except extension
        self.filenames = [f for f in os.listdir(dataset_path) if f.endswith('.txt')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        caption_file = self.filenames[idx]
        caption_path = os.path.join(self.dataset_path, caption_file)
        
        # Load caption
        with open(caption_path, 'r', encoding='utf-8') as file:
            caption = file.read().strip()

        # Assuming the image has the same name as the caption file but with a different extension (e.g., .jpg)
        image_name = caption_file.rsplit('.', 1)[0]  # Remove the .txt extension
        image_path = os.path.join(self.dataset_path, f"{image_name}.png")  # Change '.jpg' if your images have a different format

        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply the transform to convert PIL image to tensor
        image = self.transform(image)

        return image, caption
def compute_loss(generated_images, target_images):
    # Assuming generated_images and target_images are normalized in the same way
    loss = F.mse_loss(generated_images, target_images)
    return loss

image_processor =  VaeImageProcessor()

def train_ella(
    epochs=10,
    learning_rate=1e-4,
    device='cuda',
    ella_path='path_to_ella_model',
    sd_path='path_to_sdxl_model',
    dataset_path='path_to_dataset',
    t5_path='path_to_t5_model',
):
    # Load dataset
    dataset = ImageCaptionDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Load SDXL and ELLA models
    sd_model = StableDiffusionPipeline.from_pretrained(sd_path).to(device)
    ella = ELLA().to(device)
    
    # Load T5 Text Embedder
    t5_encoder = T5TextEmbedder(t5_path).to(device)

    optimizer = torch.optim.Adam(ella.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for image, caption in progress_bar:
            optimizer.zero_grad()

            image = image.to(device)  # Move image to device
            text_embedding = t5_encoder(caption).to(device)

            # Simulate timestep for the diffusion process
            timestep = torch.tensor([0.5]).to(device)  # Example timestep

            # Get conditioned embeddings from ELLA
            conditioned_embedding = ella(text_embedding, timestep)

            # Generate image based on the conditioned embeddings
            with torch.no_grad():
                generated_image = sd_model(
                    prompt_embeds=conditioned_embedding,
                    guidance_scale=7.5,
                    num_inference_steps=50,
                    height=image.shape[-2],  # Use the height of the training image
                    width=image.shape[-1],   # Use the width of the training image
                    output_type="pt"
                ).images[0]  # Get the first (and only) generated image

            # Ensure the generated image tensor requires gradients
            generated_image.requires_grad = True

            # Compute the loss between the generated image and the actual image
            loss = compute_loss(generated_image, image)

            # Backpropagate the loss
            loss.backward()

            # Update the model parameters
            optimizer.step()

            # Update the progress bar with the current loss
            progress_bar.set_postfix({"Loss": loss.item()})

        # Evaluate and save the model periodically
        if epoch % 2 == 0:
            torch.save(ella.state_dict(), f'{ella_path}_{epoch}.pth')


def validate(ella, dataloader):
    ella.eval()
    with torch.no_grad():
        for batch in dataloader:
            # Perform validation steps
            pass
    ella.train()

if __name__ == "__main__":
    train_ella(
        dataset_path="/mnt/pool/training/datasets/mj",
        ella_path="/mnt/pool/inference/ComfyUI/models/ella/ella-sd1.5-tsc-t5xl.safetensors",
        sd_path="runwayml/stable-diffusion-v1-5",
        t5_path="/mnt/pool/inference/ComfyUI/models/t5_model/flan-t5-xl"
    )