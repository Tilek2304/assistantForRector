import serial
import time

class Servo:
    def __init__(self, port, baudrate=9600):
        self.ser = None # Инициализируем self.ser как None
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Дать Arduino время перезапуститься
            print(f"Servo: Успешно подключено к {port}")
        except Exception as e:
            # Важно: Здесь мы просто печатаем ошибку.
            # Основной код (main.py) должен проверить, был ли self.ser успешно создан.
            print(f"Servo: Ошибка подключения к Arduino на {port}: {e}. Убедитесь, что Arduino подключен и порт правильный.")

    def write(self, channel: int, angle: int):
        """
        Управляет сервоприводом на заданном канале (0–15) углом (0–180)
        """
        cmd = f"SERVO:{channel}:{angle}\n"
        try:
            # Убедимся, что self.ser существует и открыт, прежде чем писать
            if hasattr(self, 'ser') and self.ser is not None and self.ser.is_open:
                self.ser.write(cmd.encode())
            # else: # Можно раскомментировать для более подробной отладки
            #     print(f"Servo: Не удалось отправить команду '{cmd.strip()}'. Соединение не установлено или закрыто.")
        except Exception as e:
            print(f"Servo: Ошибка при отправке команды '{cmd.strip()}': {e}")
            # Здесь pass может скрывать дальнейшие проблемы, но для простоты оставляем его.

    def close(self):
        # Убедимся, что self.ser существует и открыт перед закрытием
        if hasattr(self, 'ser') and self.ser is not None and self.ser.is_open:
            self.ser.close()
            print("Servo: Последовательное соединение закрыто.")
        # else: # Можно раскомментировать для более подробной отладки
        #     print("Servo: Последовательное соединение уже закрыто или не было установлено.")