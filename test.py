import torch
import torch.nn as nn
from PIL import Image
import string
import torchvision.transforms as transforms
from typing import List, Tuple
import matplotlib.pyplot as plt
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

class CaptchaTester:
    def __init__(self, model_path: str, image_height: int = 50, image_width: int = 200):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.chars = string.ascii_uppercase + string.digits
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        # Cargar modelo
        self.model = CaptchaCNN(num_chars=6, num_classes=len(self.chars)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Preprocesamiento
        self.transform = transforms.Compose([
            transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
        ])
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocesa una imagen para la inferencia"""
        image = Image.open(image_path).convert('L')
        image = self.transform(image)
        return image.unsqueeze(0).to(self.device)
    
    def predict(self, image_path: str) -> Tuple[str, List[float]]:
        """
        Predice el texto del CAPTCHA y retorna las probabilidades
        """
        # Preprocesar imagen
        image = self.preprocess_image(image_path)
        
        # Realizar predicción
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.softmax(outputs, dim=2)
            predictions = torch.argmax(outputs, dim=2)
        
        # Convertir predicciones a texto
        predicted_text = ''
        confidence_scores = []
        
        for i in range(predictions.size(1)):
            char_idx = predictions[0, i].item()
            char = self.idx_to_char[char_idx]
            predicted_text += char
            
            # Obtener probabilidad de la predicción
            confidence = probabilities[0, i, char_idx].item()
            confidence_scores.append(confidence)
        
        return predicted_text, confidence_scores
    
    def visualize_prediction(self, image_path: str, save_path: str = None):
        """
        Visualiza la imagen con la predicción y las probabilidades
        """
        # Obtener predicción
        text, confidences = self.predict(image_path)
        
        # Cargar y mostrar imagen
        image = Image.open(image_path)
        plt.figure(figsize=(12, 4))
        
        # Subplot para la imagen
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f'Predicción: {text}')
        plt.axis('off')
        
        # Subplot para las probabilidades
        plt.subplot(1, 2, 2)
        plt.bar(list(text), confidences)
        plt.title('Probabilidades por carácter')
        plt.ylim(0, 1)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def batch_test(self, image_paths: List[str]) -> List[Tuple[str, List[float]]]:
        """
        Prueba el modelo con múltiples imágenes
        """
        results = []
        for path in image_paths:
            text, confidences = self.predict(path)
            results.append((path, text, confidences))
            print(f"Imagen: {path}")
            print(f"Texto predicho: {text}")
            print(f"Confianza promedio: {sum(confidences)/len(confidences):.2%}\n")
        return results

def test_model():
    """
    Función principal para probar el modelo
    """
    # Inicializar el tester
    tester = CaptchaTester(
        model_path='best_model.pth',  # Asegúrate de que este sea el path correcto
        image_height=50,
        image_width=200
    )
    
    # Ejemplo de uso individual
    image_path = "styled_captcha_dataset/images/captcha_004000.png"
    predicted_text, confidences = tester.predict(image_path)
    print(f"Texto predicho: {predicted_text}")
    print(f"Confianzas: {[f'{conf:.2%}' for conf in confidences]}")
    
    # Visualizar predicción
    tester.visualize_prediction(image_path)
    
    # Ejemplo de prueba por lotes
    test_images = [
        "styled_captcha_dataset/images/captcha_004000.png",
        "styled_captcha_dataset/images/captcha_001000.png",
        "styled_captcha_dataset/images/captcha_003000.png"
    ]
    results = tester.batch_test(test_images)

if __name__ == "__main__":
    test_model()