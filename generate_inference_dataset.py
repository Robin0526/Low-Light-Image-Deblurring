# test_set_eval.py
# Low-Light Deblurring Test Set Evaluation Script

import os
import argparse
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import time
from tqdm import tqdm

# ----------------------------
# Model Loading
# ----------------------------
def load_model(model_type, model_path, device):
    """Load model based on model type"""
    if model_type == 'original':
        from deblur_unet_model import LowLightDeblurNet
    elif model_type == 'enhanced':
        from new_unet_model import LowLightDeblurNet
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = LowLightDeblurNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model

# ----------------------------
# Dataset Class
# ----------------------------
class LowLightDeblurDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        exts = ('*.png', '*.jpg', '*.jpeg')
        self.input_paths = []
        
        # Find all input images in low_blur_noise subdirectories
        for ext in exts:
            self.input_paths += glob(os.path.join(root_dir, '**', ext), recursive=True)

        self.input_paths.sort()
        print(f"[Dataset] Found {len(self.input_paths)} input images")

        if not self.input_paths:
            raise RuntimeError("No image files found")

        self.transform = transform

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        in_path = self.input_paths[idx]
        
        # Extract relative path from root directory
        rel_path = os.path.relpath(in_path, self.root_dir)
        folder_name = os.path.dirname(rel_path)
        file_name = os.path.basename(in_path)
        
        input_img = Image.open(in_path).convert('RGB')
        
        if self.transform:
            input_img = self.transform(input_img)
            
        return input_img, file_name, folder_name

# ----------------------------
# Dataset Generation Function
# ----------------------------
def generate_dataset(model, device, input_dir, output_dir, img_size=256):
    """Generate enhanced images for entire dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    dataset = LowLightDeblurDataset(input_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    print(f"Generating enhanced images for {len(dataset)} images...")
    
    model.eval()
    with torch.no_grad():
        for inputs, filename, folder_name in tqdm(dataloader, desc="Processing"):
            inputs = inputs.to(device)
            
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            
            # Ensure output is in [0,1] range
            outputs = torch.clamp(outputs, 0.0, 1.0)
            
            # Save enhanced image with same directory structure
            folder_path = os.path.join(output_dir, folder_name[0])
            os.makedirs(folder_path, exist_ok=True)
            
            to_pil = transforms.ToPILImage()
            output_img = to_pil(outputs.squeeze(0).cpu())
            output_img.save(os.path.join(folder_path, filename[0]))
    
    print(f"Enhanced images saved to: {output_dir}")

# ----------------------------
# Main Function
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description='Low-Light Deblurring Test Set Evaluation')
    parser.add_argument('--model_path', required=True, help='Path to trained model weights')
    parser.add_argument('--model_type', required=True, choices=['original', 'enhanced'], 
                        help='Model type: original or enhanced')
    parser.add_argument('--input_path', required=True, 
                        help='Path to test set input images (low_blur_noise directory)')
    parser.add_argument('--output_dir', required=True, 
                        help='Output directory for enhanced images')
    parser.add_argument('--img_size', type=int, default=256, 
                        help='Input image size')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Compute device (cuda/cpu)')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    print(f"Model type: {args.model_type}")
    print(f"Input directory: {args.input_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Load model
    model = load_model(args.model_type, args.model_path, args.device)
    print("Model loaded successfully")
    
    # Generate enhanced dataset
    generate_dataset(
        model=model,
        device=args.device,
        input_dir=args.input_path,
        output_dir=args.output_dir,
        img_size=args.img_size
    )
    
    print("Processing completed!")

if __name__ == '__main__':
    main()