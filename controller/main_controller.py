import threading
import os
import time
from tkinter import filedialog
from models.cnn_model import CnnModel
from models.data_genarator import DataGenerator 

class MainController:
    def __init__(self):
        self.model_gen = DataGenerator()
        self.cnn_model = CnnModel()
        self.view = None 
        
    def set_view(self, view):
        self.view = view

    # --- Генерация ---
    def on_generate_click(self):
        try:
            count = self.view.get_count_value()
        except ValueError:
            self.view.lbl_status.config(text="Ошибка: Введите число!")
            return

        self.view.lock_ui()
        thread = threading.Thread(target=self._run_generation, args=(count,))
        thread.start()

    def _run_generation(self, count):
        self.model_gen.generate_dataset(count, update_callback=self.view.update_progress)
        self.view.after(0, self.view.generation_complete)
        
    # --- Обучение ---
    def on_train_click(self):
        if not os.path.exists("data") or not os.listdir("data"):
            self.view.lbl_train_status.config(text="Ошибка: Сначала сгенерируйте данные!")
            return

        self.view.btn_train.config(state="disabled")
        self.view.lbl_train_status.config(text="Инициализация...")
        self.view.reset_training_progress(total_epochs=5)
        
        thread = threading.Thread(target=self._run_training)
        thread.start()
        
    def _run_training(self):
        def update_ui_callback(epoch, acc, loss):
            self.view.after(0, lambda: self.view.update_training_status(epoch, acc, loss))

        try:
            self.cnn_model.train(data_dir="data", epochs=5, callback_fn=update_ui_callback)
            self.view.after(0, self.view.training_complete)
        except Exception as e:
            print(f"Error inside thread: {e}")
            self.view.after(0, lambda: self.view.lbl_train_status.config(text="Ошибка обучения"))

    # --- Тестирование (Предсказание) ---
    def on_load_click(self):
        
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png;*.jpg;*.jpeg")]
        )
        if not file_path:
            return
            
        self.view.show_processing_state(file_path)
        
        thread = threading.Thread(target=self._run_prediction_process, args=(file_path,))
        thread.start()
        
    def _run_prediction_process(self, file_path):
        time.sleep(0.8) 
        
        label, prob = self.cnn_model.predict_image(file_path)
        
        self.view.after(0, lambda: self.view.show_final_result(label, prob))