import os
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# Import the PyTorch SA-UNet model
from SA_UNet_torch import create_sa_unet

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Define paths
data_location = './SA-UNet/'
training_images_loc = data_location + 'CHASE/train/image/'
training_label_loc = data_location + 'CHASE/train/label/'
validate_images_loc = data_location + 'CHASE/validate/images/'
validate_label_loc = data_location + 'CHASE/validate/labels/'
model_save_path = './SA-UNet/Model/CHASE/SA_UNet.pt' 

# Set desired size
desired_size = 512


# Create a custom dataset class
class RetinaDataset(Dataset):
    def __init__(self, images_loc, labels_loc, desired_size=512, transform=None):
        self.images_loc = images_loc
        self.labels_loc = labels_loc
        self.files = [f for f in os.listdir(images_loc) if os.path.isfile(os.path.join(images_loc, f))]
        self.desired_size = desired_size
        self.transform = transform
        
        # Validate files exist
        for img_name in self.files:
            label_name = img_name.split('_')[0] + "_" + img_name.split('_')[1].split(".")[0] + "_1stHO.png"
            label_path = os.path.join(self.labels_loc, label_name)
            if not os.path.exists(label_path):
                print(f"Warning: Label file {label_path} does not exist for image {img_name}")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            # Read image
            img_name = self.files[idx]
            img_path = os.path.join(self.images_loc, img_name)
            img = imageio.imread(img_path)
            
            # Read label
            label_name = img_name.split('_')[0] + "_" + img_name.split('_')[1].split(".")[0] + "_1stHO.png"
            label_path = os.path.join(self.labels_loc, label_name)
            label = imageio.imread(label_path, pilmode="L")
            
            # Resize images to desired size
            img = cv2.resize(img, (self.desired_size, self.desired_size))
            label = cv2.resize(label, (self.desired_size, self.desired_size))
            
            # Threshold label
            _, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)
            
            # Normalize
            img = img.astype('float32') / 255.0
            label = label.astype('float32') / 255.0
            
            # Convert to torch tensors and adjust dimensions
            img = torch.from_numpy(img).permute(2, 0, 1)  # Change from (H,W,C) to (C,H,W)
            label = torch.from_numpy(label).unsqueeze(0)  # Add channel dimension
            
            if self.transform:
                img = self.transform(img)
                
            return img, label
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            # Return a placeholder instead of crashing
            img = torch.zeros((3, self.desired_size, self.desired_size), dtype=torch.float32)
            label = torch.zeros((1, self.desired_size, self.desired_size), dtype=torch.float32)
            return img, label


# Create datasets and dataloaders
train_dataset = RetinaDataset(training_images_loc, training_label_loc, desired_size)
val_dataset = RetinaDataset(validate_images_loc, validate_label_loc, desired_size)

# Reduce num_workers to avoid multiprocessing issues
# Use persistent_workers=True to keep workers alive between iterations
train_loader = DataLoader(
    train_dataset, 
    batch_size=2, 
    shuffle=True, 
    num_workers=2,  # Reduced from 4 to 2
    persistent_workers=True if torch.cuda.is_available() else False,
    pin_memory=True if torch.cuda.is_available() else False
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=2, 
    shuffle=False, 
    num_workers=2,  # Reduced from 4 to 2
    persistent_workers=True if torch.cuda.is_available() else False,
    pin_memory=True if torch.cuda.is_available() else False
)

# Create model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create the SA-UNet model 
model, criterion, optimizer = create_sa_unet(input_size=(3, desired_size, desired_size), 
                                           block_size=7, 
                                           keep_prob=0.9, 
                                           start_neurons=16, 
                                           lr=1e-3)
model = model.to(device)

# Load weights if restore is True
restore = False
if restore and os.path.isfile(model_save_path):
    model.load_state_dict(torch.load(model_save_path))
    print(f"Loaded model weights from {model_save_path}")

# TensorBoard writer
writer = SummaryWriter('./runs/sa_unet_experiment')

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_val_acc = 0.0
    train_acc_history = []
    val_acc_history = []
    
    # Create directory for model if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_pixels = 0
        
        # Use a try-except block to catch errors during training
        try:
            for inputs, labels in tqdm(train_loader, desc="Training"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                preds = (outputs > 0.5).float()
                running_corrects += (preds == labels).sum().item()
                total_pixels += torch.numel(labels)
        
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects / total_pixels
            train_acc_history.append(epoch_acc)
            
            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Validation phase
            model.eval()
            val_running_loss = 0.0
            val_running_corrects = 0
            val_total_pixels = 0
            
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc="Validation"):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Statistics
                    val_running_loss += loss.item() * inputs.size(0)
                    preds = (outputs > 0.5).float()
                    val_running_corrects += (preds == labels).sum().item()
                    val_total_pixels += torch.numel(labels)
            
            val_epoch_loss = val_running_loss / len(val_loader.dataset)
            val_epoch_acc = val_running_corrects / val_total_pixels
            val_acc_history.append(val_epoch_acc)
            
            print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
            
            # Log to TensorBoard
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            writer.add_scalar('Loss/val', val_epoch_loss, epoch)
            writer.add_scalar('Accuracy/train', epoch_acc, epoch)
            writer.add_scalar('Accuracy/val', val_epoch_acc, epoch)
            
            # Save best model
            if val_epoch_acc > best_val_acc:
                best_val_acc = val_epoch_acc
                torch.save(model.state_dict(), model_save_path)
                print(f"Saved model with validation accuracy: {best_val_acc:.4f}")
            
            # Learning rate adjustment (optional)
            if epoch == 50:  # After 50 epochs, reduce learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-4
                print("Learning rate adjusted to 1e-4")
                
        except Exception as e:
            print(f"Error during training: {e}")
            # Save model on error to prevent losing progress
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc_history,
                'val_acc': val_acc_history
            }, model_save_path.replace('.pth', f'_error_epoch_{epoch}.pth'))
            print(f"Saved checkpoint due to error at epoch {epoch}")
            continue
    
    return model, train_acc_history, val_acc_history


if __name__ == "__main__":
    # This helps with multiprocessing issues on Windows
    # Train the model
    try:
        model, train_acc_history, val_acc_history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

        # Close TensorBoard writer
        writer.close()

        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(train_acc_history, label='Train Accuracy')
        plt.plot(val_acc_history, label='Validation Accuracy')
        plt.title('SA-UNet Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.savefig('accuracy_plot.png')
        plt.show()

        # Save final model regardless of performance
        torch.save({
            'epoch': 100,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': train_acc_history,
            'val_acc': val_acc_history
        }, model_save_path.replace('.pth', '_final.pth'))
        print(f"Saved final model checkpoint to {model_save_path.replace('.pth', '_final.pth')}")
    except Exception as e:
        print(f"Training failed with error: {e}")
