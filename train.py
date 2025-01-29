import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import string
from tqdm import tqdm
import numpy as np

class CaptchaDataset(Dataset):
    def __init__(self, labels_file, transform=None):
        self.data = pd.read_csv(labels_file, header=None, names=['image_path', 'text'])
        self.transform = transform
        self.chars = string.ascii_uppercase + string.digits
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        text = self.data.iloc[idx]['text']
        
        # Cargar y transformar imagen
        image = Image.open(img_path).convert('L')  # Convertir a escala de grises
        if self.transform:
            image = self.transform(image)
        
        # Convertir texto a tensor
        target = torch.zeros(len(text), len(self.chars))
        for i, char in enumerate(text):
            target[i][self.char_to_idx[char]] = 1
            
        return image, target

class CaptchaCNN(nn.Module):
    def __init__(self, num_chars, num_classes):
        super(CaptchaCNN, self).__init__()
        
        # Capas convolucionales
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )
        
        # Calcular tamaño de salida de las capas convolucionales
        self.conv_output_size = self._get_conv_output_size()
        
        # Capas fully connected
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_chars * num_classes)
        )
        
        self.num_chars = num_chars
        self.num_classes = num_classes
        
    def _get_conv_output_size(self):
        # Pasar un batch dummy para calcular el tamaño de salida
        x = torch.randn(1, 1, 50, 200)
        x = self.conv_layers(x)
        return x.numel() // x.size(0)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        # Reshape para obtener predicciones por carácter
        x = x.view(-1, self.num_chars, self.num_classes)
        return x

class CaptchaTrainer:
    def __init__(self, num_chars=6, image_width=200, image_height=50):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_chars = num_chars
        self.chars = string.ascii_uppercase + string.digits
        self.num_classes = len(self.chars)
        
        # Definir transformaciones
        self.transform = transforms.Compose([
            transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
        ])
        
    def train(self, dataset_path, batch_size=32, epochs=50, learning_rate=0.001):
        # Crear dataset y dataloaders
        dataset = CaptchaDataset(dataset_path, transform=self.transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Crear modelo
        model = CaptchaCNN(self.num_chars, self.num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
        
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            # Entrenamiento
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs.view(-1, self.num_classes), labels.view(-1, self.num_classes))
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Calcular accuracy
                predictions = outputs.argmax(dim=2)
                targets = labels.argmax(dim=2)
                train_correct += (predictions == targets).sum().item()
                train_total += targets.numel()
                
                pbar.set_postfix({'loss': train_loss / (pbar.n + 1),
                                'accuracy': train_correct / train_total})
            
            # Validación
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs.view(-1, self.num_classes), labels.view(-1, self.num_classes))
                    val_loss += loss.item()
                    
                    predictions = outputs.argmax(dim=2)
                    targets = labels.argmax(dim=2)
                    val_correct += (predictions == targets).sum().item()
                    val_total += targets.numel()
            
            val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / val_total
            
            print(f'\nValidation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
            
            scheduler.step(val_loss)
        
        return model

if __name__ == "__main__":
    trainer = CaptchaTrainer()
    model = trainer.train("styled_captcha_dataset/labels.csv")