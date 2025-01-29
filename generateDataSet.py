import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import random
import string
import os
from tqdm import tqdm

class StyledCaptchaGenerator:
    def __init__(self,
                 width=200,
                 height=50,
                 char_length=6):
        self.width = width
        self.height = height
        self.char_length = char_length
        self.chars = string.ascii_uppercase + string.digits  # Solo mayúsculas y números como en las imágenes
        self.font_sizes = (35, 38, 40)  # Tamaños más consistentes
        self.background_colors = [
            (200, 255, 200),  # Verde claro
            (200, 200, 255),  # Azul claro
            (255, 200, 255),  # Morado claro
        ]
        self.text_colors = [
            (0, 100, 0),      # Verde oscuro
            (0, 0, 100),      # Azul oscuro
            (100, 0, 100),    # Morado oscuro
        ]
        
        self.fonts_dir = 'fonts'
        if not os.path.exists(self.fonts_dir):
            os.makedirs(self.fonts_dir)
    
    def _create_bubble_background(self):
        """Crea un fondo con burbujas como en las imágenes de ejemplo"""
        image = Image.new('RGB', (self.width, self.height), 'white')
        draw = ImageDraw.Draw(image)
        
        # Añadir burbujas de fondo
        for _ in range(15):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            size = random.randint(10, 30)
            color = random.choice(self.background_colors)
            draw.ellipse([x, y, x + size, y + size], fill=color)
        
        return image
    
    def generate_captcha(self):
        # Crear fondo con burbujas
        image = self._create_bubble_background()
        draw = ImageDraw.Draw(image)
        
        # Generar texto
        text = ''.join(random.choices(self.chars, k=self.char_length))
        
        # Posicionar caracteres
        x_offset = 20
        for char in text:
            # Seleccionar fuente y color
            font_size = random.choice(self.font_sizes)
            font = ImageFont.truetype(self._get_random_font(), font_size)
            color = random.choice(self.text_colors)
            
            # Añadir sombra/glow
            for offset in range(2):
                draw.text((x_offset + offset, 10 + offset), char, 
                         font=font, fill=(255, 255, 255))
            
            # Dibujar carácter principal
            draw.text((x_offset, 10), char, font=font, fill=color)
            
            # Añadir burbuja detrás del carácter
            size = font_size + 10
            draw.ellipse([x_offset-5, 5, x_offset+size, size+10],
                        fill=random.choice(self.background_colors),
                        outline=None)
            
            # Redibujar carácter sobre la burbuja
            draw.text((x_offset, 10), char, font=font, fill=color)
            
            x_offset += font_size - 10
        
        return image, text

    def _get_random_font(self):
        fonts = [f for f in os.listdir(self.fonts_dir) if f.endswith('.ttf')]
        if not fonts:
            raise Exception("No se encontraron fuentes TTF en el directorio 'fonts'")
        return os.path.join(self.fonts_dir, random.choice(fonts))

def generate_dataset(num_samples, output_dir="styled_captcha_dataset"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, "images"))
    
    generator = StyledCaptchaGenerator()
    
    labels = []
    print(f"Generando {num_samples} CAPTCHAs...")
    for i in tqdm(range(num_samples)):
        image, text = generator.generate_captcha()
        image_path = os.path.join(output_dir, "images", f"captcha_{i:06d}.png")
        image.save(image_path)
        labels.append(f"{image_path},{text}")
    
    with open(os.path.join(output_dir, "labels.csv"), "w") as f:
        f.write("\n".join(labels))
    
    print(f"Dataset generado en {output_dir}")

if __name__ == "__main__":
    generate_dataset(15000)