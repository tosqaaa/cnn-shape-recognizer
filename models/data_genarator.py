import os
import random
import shutil
from PIL import Image, ImageDraw

class DataGenerator:
    def __init__(self, base_dir="data", image_size=(64, 64)):
        self.base_dir = base_dir
        self.image_size = image_size
        self.classes = ["circles", "squares"]

    def prepare_directories(self):
        """Очищает и создает папки для датасета."""
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)
        
        for cls in self.classes:
            os.makedirs(os.path.join(self.base_dir, cls))

    def generate_dataset(self, count_per_class, update_callback=None):
        """
        Генерирует изображения.
        update_callback - функция, которую дергаем для обновления прогресса в UI
        """
        self.prepare_directories()
        total_images = count_per_class * 2
        
        for i in range(1, count_per_class + 1):
            # 1. Рисуем Круг
            self._draw_shape("circles", i)
            if update_callback: update_callback(i * 2 + 1, total_images)

            # 2. Рисуем Квадрат
            self._draw_shape("squares", i)
            if update_callback: update_callback(i * 2 + 2, total_images)

    def _draw_shape(self, shape_type, index):
        # Создаем белое полотно (L - черно-белое, RGB - цветное)
        img = Image.new("L", self.image_size, "white") 
        draw = ImageDraw.Draw(img)
        
        w, h = self.image_size
        # Случайный размер фигуры (от 10 до 40 пикселей)
        shape_w = random.randint(10, w // 2)
        shape_h = shape_w # Для круга и квадрата ширина = высота
        
        # Случайная позиция, чтобы фигура не вылезала за края
        x0 = random.randint(0, w - shape_w)
        y0 = random.randint(0, h - shape_h)
        x1 = x0 + shape_w
        y1 = y0 + shape_h
        
        if shape_type == "circles":
            # outline - цвет контура (0 - черный), width - толщина
            draw.ellipse([x0, y0, x1, y1], outline=0, width=2)
        elif shape_type == "squares":
            draw.rectangle([x0, y0, x1, y1], outline=0, width=2)
            
        # Сохранение
        path = os.path.join(self.base_dir, shape_type, f"{index}.png")
        img.save(path)