import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class MainView(tk.Tk):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.title("CNN Shapes Recognizer")
        
        self.geometry("400x600")
        
        self._setup_ui()

    def _setup_ui(self):
        # --- Блок Генерации ---
        frame_gen = ttk.LabelFrame(self, text="1. Генерация Датасета")
        frame_gen.pack(pady=5, padx=10, fill="x")

        frame_top = ttk.Frame(frame_gen)
        frame_top.pack(fill="x", pady=5)
        ttk.Label(frame_top, text="Кол-во:").pack(side="left", padx=5)
        self.entry_count = ttk.Entry(frame_top, width=10)
        self.entry_count.insert(0, "200")
        self.entry_count.pack(side="left", padx=5)
        self.btn_generate = ttk.Button(frame_top, text="Пуск", command=self.controller.on_generate_click)
        self.btn_generate.pack(side="left", padx=5)
        
        self.progress_bar = ttk.Progressbar(frame_gen, orient="horizontal", mode="determinate")
        self.progress_bar.pack(pady=5, padx=10, fill="x")
        self.lbl_status = ttk.Label(frame_gen, text="Ожидание")
        self.lbl_status.pack(pady=2)

        # --- Блок Обучения ---
        frame_train = ttk.LabelFrame(self, text="2. Обучение Модели")
        frame_train.pack(pady=5, padx=10, fill="x")

        self.btn_train = ttk.Button(frame_train, text="Обучить Нейросеть", command=self.controller.on_train_click)
        self.btn_train.pack(pady=5)
        
        self.progress_train = ttk.Progressbar(frame_train, orient="horizontal", mode="determinate")
        self.progress_train.pack(pady=5, padx=10, fill="x")

        self.lbl_train_status = ttk.Label(frame_train, text="Статус: Модель не обучена")
        self.lbl_train_status.pack(pady=2)
        self.lbl_accuracy = ttk.Label(frame_train, text="Точность: -")
        self.lbl_accuracy.pack(pady=2)

        # --- Блок Тестирования ---
        frame_test = ttk.LabelFrame(self, text="3. Тестирование")
        frame_test.pack(pady=10, padx=10, fill="both", expand=True)

        self.btn_load = ttk.Button(frame_test, text="Загрузить и Проверить", command=self.controller.on_load_click)
        self.btn_load.pack(pady=10)
        
        # Контейнер для картинки с рамкой
        self.frame_preview = tk.Frame(frame_test, bg="gray", bd=2)
        self.frame_preview.pack(pady=5)
        
        self.lbl_image_preview = ttk.Label(self.frame_preview, text="Нет изображения")
        self.lbl_image_preview.pack(padx=2, pady=2)

        self.lbl_result = ttk.Label(frame_test, text="Результат: -", font=("Arial", 12, "bold"))
        self.lbl_result.pack(pady=10)

    
    def show_processing_state(self, image_path):
        """Показывает картинку и статус 'Анализ...'"""
        pil_image = Image.open(image_path)
        pil_image = pil_image.resize((120, 120))
        tk_image = ImageTk.PhotoImage(pil_image)
        
        self.lbl_image_preview.config(image=tk_image, text="")
        self.lbl_image_preview.image = tk_image 

        self.frame_preview.config(bg="gray")
        
        # Статус
        self.lbl_result.config(text="Анализ изображения...", foreground="black")
        self.btn_load.config(state="disabled")
        self.update_idletasks()

    def show_final_result(self, result_text, confidence):
        """Показывает финальный результат с цветной рамкой"""
        
        if "Квадрат" in result_text:
            color = "#2ecc71" # Зеленый
            text_res = "ЭТО КВАДРАТ"
        elif "Окружность" in result_text:
            color = "#3498db" # Синий
            text_res = "ЭТО ОКРУЖНОСТЬ"
        else:
            color = "red"
            text_res = "ОШИБКА"

        self.frame_preview.config(bg=color)
        
        self.lbl_result.config(
            text=f"{text_res}\nВероятность: {confidence:.1%}",
            foreground=color
        )
        self.btn_load.config(state="normal")

    # --- Генерация/Обучение ---
    def get_count_value(self):
        return int(self.entry_count.get())

    def update_progress(self, current, total):
        self.progress_bar["maximum"] = total
        self.progress_bar["value"] = current
        self.lbl_status.config(text=f"Генерация: {current}/{total}")
        self.update_idletasks()

    def generation_complete(self):
        self.lbl_status.config(text="Готово! Датасет создан.")
        self.btn_generate.config(state="normal")

    def lock_ui(self):
        self.btn_generate.config(state="disabled")

    def reset_training_progress(self, total_epochs):
        self.progress_train["maximum"] = total_epochs
        self.progress_train["value"] = 0
        self.lbl_accuracy.config(text="Точность: -")

    def update_training_status(self, epoch, acc, loss):
        self.lbl_train_status.config(text=f"Обучение... Эпоха {epoch}")
        self.lbl_accuracy.config(text=f"Acc: {acc:.2f} | Loss: {loss:.4f}")
        self.progress_train["value"] = epoch

    def training_complete(self):
        self.lbl_train_status.config(text="Обучение завершено!")
        self.progress_train["value"] = self.progress_train["maximum"]
        self.btn_train.config(state="normal")