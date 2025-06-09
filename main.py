import time
import cv2
import numpy as np
from gpt4all import GPT4All
import threading
from servo import Servo # Импортируем ваш класс Servo
import configparser # Для чтения конфигурационного файла

# --- Для STT (распознавание речи) ---
import speech_recognition as sr

# --- Для TTS (синтез речи) ---
from gtts import gTTS
import os
import pygame # Для воспроизведения аудио
import io # Для работы с аудио в памяти




# --- Глобальные переменные ---
is_true = False # Флаг для активации маха рукой
num_faces = 0   # Количество обнаруженных лиц
lock = threading.Lock() # Блокировка для управления доступом к общим переменным
robot_name = "Бот" # Имя робота по умолчанию
robot_developer = "Неизвестно" # Разработчик по умолчанию


# --- Режимы ---
testing = False

# --- Инициализация Pygame Mixer для воспроизведения аудио ---
pygame.mixer.init()

# --- Конфигурация из файла ---
config = configparser.ConfigParser()
try:
    config.read('config.ini')
    # Информация о боте
    robot_name = config.get('BOT_INFO', 'name', fallback='Бот')
    robot_developer = config.get('BOT_INFO', 'developer', fallback='Неизвестно')
    creation_date = config.get('BOT_INFO', 'creation_date', fallback='Неизвестно')
    description = config.get('BOT_INFO', 'description', fallback='Нейрочат-робот.')
    print(f"Конфигурация: Имя робота - {robot_name}, Разработчик - {robot_developer}")

    # Параметры последовательного порта
    SERIAL_PORT = config.get('SERIAL_PORT', 'port', fallback='COM3')
    BAUDRATE = config.getint('SERIAL_PORT', 'baudrate', fallback=9600)
    print(f"Конфигурация: Порт - {SERIAL_PORT}, Скорость - {BAUDRATE}")

    # Параметры TTS
    TTS_LANGUAGE = config.get('TTS', 'language', fallback='ru')
    print(f"Конфигурация: Язык TTS - {TTS_LANGUAGE}")

    # Параметры STT
    MICROPHONE_INDEX = config.getint('STT', 'microphone_index', fallback=0)
    PHRASE_TIME_LIMIT = config.getint('STT', 'phrase_time_limit', fallback=5)
    print(f"Конфигурация: Индекс микрофона - {MICROPHONE_INDEX}, Лимит фразы - {PHRASE_TIME_LIMIT}с")

except Exception as e:
    print(f"Ошибка чтения config.ini: {e}. Используются значения по умолчанию.")
    # Устанавливаем значения по умолчанию, если файл не найден или некорректен
    SERIAL_PORT = 'COM3'
    BAUDRATE = 9600
    TTS_LANGUAGE = 'ru'
    MICROPHONE_INDEX = 0
    PHRASE_TIME_LIMIT = 5


# --- Параметры для маха рукой ---
WAVE_UP_ANGLE = 120  # Угол для движения руки вверх
WAVE_DOWN_ANGLE = 60 # Угол для движения руки вниз


# --- Инициализация GPT4All ---
try:
    model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")
    print("GPT4All: Модель успешно загружена.")
except Exception as e:
    print(f"FATAL ERROR: Не удалось загрузить модель GPT4All: {e}")
    print("Убедитесь, что 'Meta-Llama-3-8B-Instruct.Q4_0.gguf' находится в правильной директории.")
    exit() # Завершаем программу, если модель не загружена

# --- Инициализация сервоприводов ---
try:
    controller = Servo(SERIAL_PORT, BAUDRATE) # Используем порт и скорость из конфига
    # Дополнительная проверка, чтобы убедиться, что serial.Serial был успешно создан
    if not hasattr(controller, 'ser') or not controller.ser.is_open:
        raise Exception("Не удалось установить последовательное соединение с Arduino.")
    print("Серво: Контроллер сервоприводов успешно подключен.")
except Exception as e:
    print(f"FATAL ERROR: Не удалось подключиться к контроллеру сервоприводов. {e}")
    print("Пожалуйста, проверьте последовательный порт в config.ini и убедитесь, что Arduino подключен и запущен правильный скетч.")
    if not testing:

        exit() # Завершаем программу, если не удалось подключиться к серво

SERVO_X = 0  # канал для оси X
SERVO_Y = 1  # канал для оси Y
SERVO_H = 2  # канал для руки
# Начальные углы
angle_x = 90
angle_y = 90
angle_h = 90 # Изначальный угол руки
controller.write(SERVO_X, angle_x)
controller.write(SERVO_Y, angle_y)
controller.write(SERVO_H, angle_h)

# --- Параметры камеры и отслеживания ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CENTER_X = FRAME_WIDTH // 2
CENTER_Y = FRAME_HEIGHT // 2
SENSITIVITY = 20  # пикселей
STEP = 2          # шаг изменения угла

# Загрузка классификатора лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("FATAL ERROR: Не удалось загрузить каскадный классификатор лиц. Проверьте путь.")
    exit() # Завершаем программу, если классификатор не загружен

# Захват видео
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("FATAL ERROR: Не удалось открыть видеопоток с камеры.")
    exit() # Завершаем программу, если камера недоступна
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# --- Функции TTS и STT ---

# Функция для озвучивания текста
def speak(text):
    tts = gTTS(text=text, lang=TTS_LANGUAGE, slow=False)
    # Сохраняем аудио в памяти вместо файла
    fp = io.BytesIO()
    tts.save(fp)
    fp.seek(0) # Перематываем в начало

    # Воспроизводим аудио из памяти
    try:
        pygame.mixer.music.load(fp, 'mp3') # Указываем формат mp3
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10) # Ограничиваем FPS для экономии CPU
    except pygame.error as e:
        print(f"Ошибка воспроизведения аудио: {e}. Возможно, не найден FFmpeg или проблема с форматом.")

# Функция для распознавания речи
def recognize_speech_from_mic(recognizer, microphone):
    """
    Распознает речь с микрофона.
    Возвращает текст или None в случае ошибки/неразборчивости.
    """
    with microphone as source:
        print("STT: Слушаю...")
        recognizer.adjust_for_ambient_noise(source) # Адаптация к шуму
        try:
            audio = recognizer.listen(source, phrase_time_limit=PHRASE_TIME_LIMIT)
        except sr.WaitTimeoutError:
            print("STT: Превышен лимит ожидания фразы.")
            return None
    try:
        # Используем Google Speech Recognition (требует интернет)
        text = recognizer.recognize_google(audio, language=TTS_LANGUAGE)
        print(f"STT: Вы сказали: {text}")
        return text
    except sr.UnknownValueError:
        print("STT: Не удалось распознать речь.")
        return None
    except sr.RequestError as e:
        print(f"STT: Ошибка сервиса распознавания речи; {e}")
        return None

# --- Основные функции потоков ---

def chating():
    global is_true
    global angle_h
    global testing
    r = sr.Recognizer()
    # mic = sr.Microphone() # Используем микрофон по умолчанию

    # Попробуйте найти микрофон по индексу из конфига
    try:
        mic = sr.Microphone(device_index=MICROPHONE_INDEX)
    except Exception as e:
        print(f"FATAL ERROR: Не удалось найти микрофон по индексу {MICROPHONE_INDEX}: {e}")
        print("Пожалуйста, проверьте 'microphone_index' в config.ini или используйте sr.Microphone.list_microphone_names() для поиска.")
        if not testing:
            exit()
        else:
            pass

    with model.chat_session():
        running = True
        print(f"\n--- Чат-сессия запущена. Я {robot_name}. ---")
        speak(f"Привет! Я {robot_name}. Готов к общению.")
        print("Напишите 'конец сессии' или скажите 'пока' для завершения.")

        while running:
            # Проверяем флаг для маха рукой
            if is_true:
                with lock:
                    print("Чат: Машу рукой!")
                    for j in range(3):
                        for i in range(angle_h, WAVE_UP_ANGLE + 1, STEP):
                            controller.write(SERVO_H, i)
                            time.sleep(0.015)
                        for i in range(WAVE_UP_ANGLE, WAVE_DOWN_ANGLE - 1, -STEP):
                            controller.write(SERVO_H, i)
                            time.sleep(0.015)
                        for i in range(WAVE_DOWN_ANGLE, angle_h + 1, STEP):
                            controller.write(SERVO_H, i)
                            time.sleep(0.015)
                    is_true = False # Сбрасываем флаг после маха

            # Получаем ввод: сначала пытаемся распознать голос, затем спрашиваем текст
            user_input = recognize_speech_from_mic(r, mic)
            if user_input is None:
                # Если голос не распознан, переходим к текстовому вводу
                user_input = input(">> ")

            message = user_input.strip()

            if message.lower() == 'конец сессии' or message.lower() == 'пока':
                speak("До свидания!")
                running = False
            elif message: # Только если сообщение не пустое
                print(f"GPT: Получаю ответ...")
                response = model.generate(message, max_tokens=1024)
                print(f"GPT: {response}")
                speak(response)

def tracking():
    global num_faces
    global is_true
    global angle_x, angle_y

    previous_num_faces = 0
    initial_detection = True

    print("--- Отслеживание лиц запущено ---")
    print("Нажмите 'q' в окне видео, чтобы остановить.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Отслеживание: Ошибка чтения кадра с камеры. Завершение отслеживания.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        current_num_faces = len(faces)

        with lock:
            if initial_detection:
                previous_num_faces = current_num_faces
                if previous_num_faces > 0:
                    print(f"Отслеживание: Обнаружено {previous_num_faces} лиц(а) при запуске.")
                initial_detection = False
            else:
                if current_num_faces > previous_num_faces:
                    print('Отслеживание: Привет, новый кожанный мешок!')
                    is_true = True
                elif current_num_faces < previous_num_faces:
                    print(f"Отслеживание: Количество лиц уменьшилось до {current_num_faces}.")
                elif current_num_faces == 0 and previous_num_faces > 0:
                    print("Отслеживание: Все лица исчезли.")

            num_faces = current_num_faces

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_cx = x + w // 2
            face_cy = y + h // 2

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.line(frame, (CENTER_X, CENTER_Y), (face_cx, face_cy), (255, 0, 0), 2)

            dx = face_cx - CENTER_X
            dy = face_cy - CENTER_Y

            if abs(dx) > SENSITIVITY:
                angle_x -= STEP if dx > 0 else -STEP
                angle_x = max(0, min(180, angle_x))
                controller.write(SERVO_X, angle_x)

            if abs(dy) > SENSITIVITY:
                angle_y += STEP if dy > 0 else -STEP
                angle_y = max(0, min(180, angle_y))
                controller.write(SERVO_Y, angle_y)

        cv2.imshow('Face Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Отслеживание: Нажата 'q'. Завершение потока отслеживания.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    starttestingchat = 'testchat'
    mess = input()
    if mess == starttestingchat:
        testing = True
        print(f"--- Запуск в тест режиме {robot_name} ({description}) ---")
        print(f"Разработчик: {robot_developer}, Дата создания: {creation_date}")
        chating_thread = threading.Thread(target=chating)   
        chating_thread.start()
        chating_thread.join()
        pygame.mixer.quit() # Выключаем pygame mixer
        print("Тестирование завершенно.")
        exit()

    print(f"--- Запуск {robot_name} ({description}) ---")
    print(f"Разработчик: {robot_developer}, Дата создания: {creation_date}")

    # Создаем потоки
    chating_thread = threading.Thread(target=chating)
    tracking_thread = threading.Thread(target=tracking)

    # Запускаем потоки
    chating_thread.start()
    tracking_thread.start()

    # Ожидаем завершения потоков
    chating_thread.join()
    tracking_thread.join()

    # Закрываем последовательное соединение, когда программа полностью завершается
    if controller:
        controller.close()

    pygame.mixer.quit() # Выключаем pygame mixer
    print("Программа завершена.")