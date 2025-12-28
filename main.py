from controller.main_controller import MainController
from view.main_window import MainView

# --- Запуск приложения ---
if __name__ == "__main__":
    controller = MainController()
    app = MainView(controller)
    controller.set_view(app)
    
    app.mainloop()