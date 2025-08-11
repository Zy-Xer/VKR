import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, UnidentifiedImageError
import pytesseract
import os
import cv2
import numpy as np
import requests
import json

# для установки нужных библиотек введите команду:
# pip install -r requirements.txt

class ChuvashOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Распознаватель и переводчик чувашского текста") 
        self.root.configure(bg="#fff5e0")

        # --- Конфигурация Yandex Cloud Translate API ---
        self.yandex_cloud_api_token = ""
        self.yandex_cloud_folder_id = ""
        # --- Конец конфигурации Yandex Cloud Translate API ---

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(4, weight=2)  # OCR text box
        self.root.grid_rowconfigure(6, weight=2)  # Translation text box
        self.root.grid_columnconfigure(0, weight=1)

        self.canvas_frame = tk.Frame(self.root, bg="#fff5e0")
        self.canvas_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 0))
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#f0f0f0", highlightthickness=1, highlightbackground="gray")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.label = tk.Label(self.root, text="Файл не выбран", font=("Arial", 12, "bold"),
                              bg="#fff5e0", fg="#a80000")
        self.label.grid(row=1, column=0, sticky="n", pady=(5, 5))

        self.char_frame = tk.Frame(self.root, bg="#fff5e0")
        self.char_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        chuvash_letters = ['Ӑ', 'ӑ', 'Ӗ', 'ӗ', 'Ҫ', 'ҫ', 'Ӳ', 'ӳ']
        for idx, ch in enumerate(chuvash_letters):
            b = tk.Button(self.char_frame, text=ch, font=("Arial", 11, "bold"),
                          bg="#ffd54f", fg="#5a0000", width=3, height=1,
                          relief=tk.RAISED, borderwidth=2, command=lambda c=ch: self.insert_letter(c))
            b.grid(row=0, column=idx, padx=3, pady=3)
        for i in range(len(chuvash_letters)):
            self.char_frame.grid_columnconfigure(i, weight=1)

        self.ocr_label = tk.Label(self.root, text="Распознанный текст (чувашский):", font=("Arial", 10), bg="#fff5e0")
        self.ocr_label.grid(row=3, column=0, sticky="sw", padx=15, pady=(10, 0))

        self.text_box = tk.Text(self.root, height=8, font=("Arial", 14),
                                bg="#fffdf0", fg="#333333", relief=tk.SOLID, borderwidth=1, wrap=tk.WORD,
                                insertbackground="#5a0000")
        self.text_box.grid(row=4, column=0, sticky="nsew", padx=15, pady=(0, 5))

        self.translation_label = tk.Label(self.root, text="Перевод (русский):", font=("Arial", 10),
                                          bg="#fff5e0")
        self.translation_label.grid(row=5, column=0, sticky="sw", padx=15, pady=(5, 0))

        self.translation_box = tk.Text(self.root, height=8, font=("Arial", 14),
                                       bg="#f0fff0", fg="#333333", relief=tk.SOLID, borderwidth=1, wrap=tk.WORD,
                                       state=tk.DISABLED)
        self.translation_box.grid(row=6, column=0, sticky="nsew", padx=15, pady=(0, 10))

        self.button_frame = tk.Frame(self.root, bg="#fff5e0")
        self.button_frame.grid(row=7, column=0, pady=(5, 15))

        btn_common_config = {"font": ("Arial", 11, "bold"), "relief": tk.RAISED,
                             "borderwidth": 2, "padx": 10, "pady": 5}

        self.load_button = tk.Button(self.button_frame, text="📂 Загрузить", command=self.load_image,
                                     bg="#ffd54f", fg="#5a0000", activebackground="#ffca28", **btn_common_config)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.recognize_button = tk.Button(self.button_frame, text="🔍 Распознать", command=self.recognize_text,
                                          bg="#ef5350", fg="white", activebackground="#e53935",
                                          state=tk.DISABLED, **btn_common_config)
        self.recognize_button.pack(side=tk.LEFT, padx=5)

        self.translate_button = tk.Button(self.button_frame, text="🔄 Перевести",
                                          command=self.translate_text_action,
                                          bg="#2196f3", fg="white", activebackground="#1976d2",
                                          state=tk.DISABLED, **btn_common_config)
        self.translate_button.pack(side=tk.LEFT, padx=5)

        self.save_button = tk.Button(self.button_frame, text="💾 Сохранить", command=self.save_text,
                                     bg="#4caf50", fg="white", activebackground="#43a047",
                                     state=tk.DISABLED, **btn_common_config)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(self.button_frame, text="🧹 Очистить всё", command=self.clear_all_fields,
                                      bg="#9e9e9e", fg="white", activebackground="#757575", **btn_common_config)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.image_path = None
        self.tk_image = None
        self.original_pil_image = None

        print(f"DEBUG: API Token = '{self.yandex_cloud_api_token}'")  # Отладочная печать
        print(f"DEBUG: Folder ID = '{self.yandex_cloud_folder_id}'")  # Отладочная печать

        self.is_yandex_cloud_configured = False

        placeholder_api_token = "YOUR_YANDEX_API_TOKEN_PLACEHOLDER"  # Более общий плейсхолдер
        placeholder_folder_id = "YOUR_YANDEX_FOLDER_ID_PLACEHOLDER"

        if self.yandex_cloud_api_token == placeholder_api_token or \
                self.yandex_cloud_folder_id == placeholder_folder_id or \
                not self.yandex_cloud_api_token or \
                not self.yandex_cloud_folder_id:
            print("ПРЕДУПРЕЖДЕНИЕ: API-токен или FOLDER_ID для Yandex Cloud Translate не настроены корректно.")
            messagebox.showwarning("Конфигурация Yandex Cloud API",
                                   "API-токен или FOLDER_ID для Yandex Cloud Translate не настроены в коде.\n"
                                   "Функция перевода будет недоступна.\n"
                                   "Пожалуйста, укажите их в переменных 'self.yandex_cloud_api_token' и 'self.yandex_cloud_folder_id'.")
            self.translate_button.config(state=tk.DISABLED)
        else:
            self.is_yandex_cloud_configured = True
            print("Параметры Yandex Cloud Translate API установлены.")

    def insert_letter(self, letter):
        self.text_box.insert(tk.INSERT, letter)
        self.text_box.focus_set()

    def _update_image_on_canvas(self, pil_image_to_display):
        self.root.update_idletasks()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 600, 250
        img_w, img_h = pil_image_to_display.size
        aspect_ratio = img_w / img_h
        if img_w > canvas_width or img_h > canvas_height:
            if canvas_width / aspect_ratio <= canvas_height:
                new_w = canvas_width
                new_h = int(new_w / aspect_ratio)
            else:
                new_h = canvas_height
                new_w = int(new_h * aspect_ratio)
            pil_image_to_display = pil_image_to_display.resize((new_w, new_h), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(pil_image_to_display)
        self.canvas.delete("all")
        x_center = (self.canvas.winfo_width() - self.tk_image.width()) // 2
        y_center = (self.canvas.winfo_height() - self.tk_image.height()) // 2
        self.canvas.create_image(max(0, x_center), max(0, y_center), anchor=tk.NW, image=self.tk_image)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Файлы изображений", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"), ("Все файлы", "*.*")])
        if file_path:
            try:
                self.image_path = file_path
                pil_image = Image.open(file_path)
                self.original_pil_image = pil_image.copy()
                self._update_image_on_canvas(self.original_pil_image)
                self.label.config(text=os.path.basename(file_path), fg="#00695c")
                self.text_box.delete(1.0, tk.END)
                self.translation_box.config(state=tk.NORMAL)
                self.translation_box.delete(1.0, tk.END)
                self.translation_box.config(state=tk.DISABLED)
                self.recognize_button.config(state=tk.NORMAL)
                self.save_button.config(state=tk.DISABLED)
                if self.is_yandex_cloud_configured:
                    self.translate_button.config(state=tk.DISABLED)  # Деактивируем, пока нет текста
                else:
                    self.translate_button.config(state=tk.DISABLED)

            except FileNotFoundError:
                messagebox.showerror("Ошибка", f"Файл не найден: {file_path}")
                self._reset_ui_for_new_image()
            except UnidentifiedImageError:
                messagebox.showerror("Ошибка формата",
                                     f"Не удалось распознать файл как изображение: {os.path.basename(file_path)}\n"
                                     "Пожалуйста, выберите файл формата PNG, JPG, BMP или TIFF.")
                self._reset_ui_for_new_image()
            except Exception as e:
                messagebox.showerror("Ошибка загрузки", f"Произошла непредвиденная ошибка при загрузке: {e}")
                self._reset_ui_for_new_image()

    def _reset_ui_for_new_image(self):
        self.image_path = None
        self.tk_image = None
        self.original_pil_image = None
        self.canvas.delete("all")
        self.label.config(text="Файл не выбран", fg="#a80000")
        self.text_box.delete(1.0, tk.END)
        self.translation_box.config(state=tk.NORMAL)
        self.translation_box.delete(1.0, tk.END)
        self.translation_box.config(state=tk.DISABLED)
        self.recognize_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.translate_button.config(state=tk.DISABLED)

    def preprocess_image_for_ocr(self, pil_img_original):
        try:
            pil_img_for_processing = pil_img_original.copy()
            if pil_img_for_processing.mode != 'L':
                pil_img_for_processing = pil_img_for_processing.convert('L')
            cv_image = np.array(pil_img_for_processing)
            _, thresh_image = cv2.threshold(cv_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh_image
        except cv2.error as ocve:
            messagebox.showerror("Ошибка предобработки OpenCV",
                                 f"Ошибка OpenCV при предобработке изображения: {ocve}\n"
                                 "Убедитесь, что файл изображения не поврежден и имеет стандартный формат (PNG, JPG).")
            return None
        except Exception as e:
            messagebox.showerror("Ошибка предобработки", f"Ошибка при предобработке изображения: {e}")
            return None

    def recognize_text(self):
        if not self.original_pil_image:
            messagebox.showinfo("Информация", "Пожалуйста, сначала загрузите изображение.")
            return

        self.label.config(text="Распознавание...", fg="#0d47a1")
        self.root.update_idletasks()
        e_local = None

        try:
            processed_image_cv = self.preprocess_image_for_ocr(self.original_pil_image)
            if processed_image_cv is None:
                self.label.config(text=os.path.basename(self.image_path) if self.image_path else "Файл не выбран",
                                  fg="#a80000")
                return

            text = pytesseract.image_to_string(processed_image_cv, lang="chv")
            self.text_box.delete(1.0, tk.END)
            self.translation_box.config(state=tk.NORMAL)
            self.translation_box.delete(1.0, tk.END)
            self.translation_box.config(state=tk.DISABLED)

            if text.strip():
                self.text_box.insert(tk.END, text.strip())
                self.label.config(text="Распознавание завершено", fg="#00695c")
                self.save_button.config(state=tk.NORMAL)
                if self.is_yandex_cloud_configured:
                    self.translate_button.config(state=tk.NORMAL)
                else:
                    self.translate_button.config(state=tk.DISABLED)
            else:
                messagebox.showwarning("Результат", "Не удалось распознать текст на изображении.\n"
                                                    "Попробуйте изображение лучшего качества или с более четким текстом.")
                self.label.config(text="Текст не распознан", fg="#a80000")
                self.save_button.config(state=tk.DISABLED)
                self.translate_button.config(state=tk.DISABLED)

        except pytesseract.TesseractNotFoundError:
            e_local = pytesseract.TesseractNotFoundError
            messagebox.showerror("Ошибка Tesseract",
                                 "Tesseract OCR не найден. Убедитесь, что он установлен и добавлен в PATH, или укажите путь к tesseract.exe в коде.")
            self.label.config(text="Ошибка: Tesseract не найден", fg="#a80000")
            self.recognize_button.config(state=tk.DISABLED)
            self.translate_button.config(state=tk.DISABLED)
        except pytesseract.TesseractError as te:
            e_local = te
            messagebox.showerror("Ошибка Tesseract",
                                 f"Произошла ошибка Tesseract: {te}\nВозможно, отсутствует языковой пакет 'chv' для Tesseract.")
            self.label.config(text="Ошибка модели 'chv'", fg="#a80000")
            self.translate_button.config(state=tk.DISABLED)
        except Exception as e:
            e_local = e
            messagebox.showerror("Ошибка распознавания", f"Произошла непредвиденная ошибка: {e}")
            self.label.config(text="Ошибка при распознавании", fg="#a80000")
            self.translate_button.config(state=tk.DISABLED)
        finally:
            if self.image_path and not isinstance(e_local, pytesseract.TesseractNotFoundError):
                current_text_status = self.label.cget("text")
                if "Распознавание завершено" not in current_text_status and \
                        "Текст не распознан" not in current_text_status and \
                        "Ошибка" not in current_text_status:
                    self.label.config(text=os.path.basename(self.image_path), fg="#00695c")

    def translate_text_action(self):
        original_text = self.text_box.get("1.0", tk.END).strip()
        if not original_text:
            messagebox.showinfo("Информация", "Нет текста для перевода.")
            return

        if not self.is_yandex_cloud_configured:
            messagebox.showerror("Ошибка конфигурации",
                                 "API-токен или FOLDER_ID для Yandex Cloud Translate не настроены.\n"
                                 "Перевод невозможен.")
            self.label.config(text="Ошибка конфиг. YC API", fg="#a80000")
            return

        self.label.config(text="Перевод (Yandex Cloud)...", fg="#0d47a1")
        self.root.update_idletasks()

        try:
            url = "https://translate.api.cloud.yandex.net/translate/v2/translate"

            # Определяем тип авторизации
            # IAM-токены обычно начинаются с "t1."
            # API-ключи (статические ключи доступа) начинаются с "AQVN"
            auth_header_value = ""
            if self.yandex_cloud_api_token.startswith("t1."):
                auth_header_value = f"Bearer {self.yandex_cloud_api_token}"
            elif self.yandex_cloud_api_token.startswith("AQVN"):  # Обычный формат API-ключа
                auth_header_value = f"Api-Key {self.yandex_cloud_api_token}"
            else:  # Если формат не ясен, предполагаем, что это API-ключ без префикса
                # Или можно вывести ошибку, если формат токена неизвестен
                messagebox.showwarning("Формат токена",
                                       "Не удалось однозначно определить тип токена (API-ключ или IAM-токен). Попытка использовать как Api-Key.")
                auth_header_value = f"Api-Key {self.yandex_cloud_api_token}"

            headers = {
                "Content-Type": "application/json",
                "Authorization": auth_header_value
            }
            body = {
                "folderId": self.yandex_cloud_folder_id,
                "sourceLanguageCode": "cv",
                "targetLanguageCode": "ru",
                "texts": [original_text]  # API ожидает список текстов
            }

            response = requests.post(url, headers=headers, json=body, timeout=15)
            response.raise_for_status()  # Проверка на HTTP ошибки (4xx, 5xx)

            response_data = response.json()
            translations = response_data.get("translations")

            if translations and len(translations) > 0 and "text" in translations[0]:
                translated_text = translations[0]["text"]
            else:
                # Попытка извлечь сообщение об ошибке из ответа API, если структура не та
                error_detail = response_data.get("message", "Ответ API не содержит ожидаемых данных о переводе.")
                raise ValueError(f"Ошибка ответа API: {error_detail}")

            self.translation_box.config(state=tk.NORMAL)
            self.translation_box.delete(1.0, tk.END)
            self.translation_box.insert(tk.END, translated_text)
            self.translation_box.config(state=tk.DISABLED)

            self.label.config(text="Перевод (YC) завершен", fg="#00695c")

        except requests.exceptions.HTTPError as http_err:
            error_message = f"Ошибка HTTP при запросе к Yandex Cloud: {http_err}\n"
            try:
                # Попытка получить детали ошибки из JSON ответа
                error_details = http_err.response.json()
                api_message = error_details.get('message', 'Нет дополнительной информации от API.')
                api_code = error_details.get('code', '')
                error_message += f"Код: {api_code}, Сообщение от API: {api_message}"
            except json.JSONDecodeError:
                error_message += f"Тело ответа (не JSON): {http_err.response.text}"
            except AttributeError:  # Если response не имеет json()
                error_message += f"Ответ сервера: {http_err.response.status_code}"
            messagebox.showerror("Ошибка перевода (YC HTTP)", error_message)
            self.label.config(text="Ошибка перевода (YC HTTP)", fg="#a80000")
        except requests.exceptions.ConnectionError:
            messagebox.showerror("Ошибка перевода (YC)", "Ошибка соединения. Проверьте подключение к интернету.")
            self.label.config(text="Ошибка соединения (YC)", fg="#a80000")
        except requests.exceptions.Timeout:
            messagebox.showerror("Ошибка перевода (YC)", "Тайм-аут запроса к Yandex Cloud. Попробуйте позже.")
            self.label.config(text="Тайм-аут (YC)", fg="#a80000")
        except ValueError as ve:  # Для наших кастомных ошибок с ответом API
            messagebox.showerror("Ошибка перевода (YC)", f"Проблема с ответом от Yandex Cloud: {ve}")
            self.label.config(text="Ошибка ответа API (YC)", fg="#a80000")
        except Exception as e:
            messagebox.showerror("Ошибка перевода (YC)",
                                 f"Не удалось перевести текст через Yandex Cloud: {type(e).__name__}: {e}\n"
                                 "Проверьте настройки API, токен, Folder ID и доступ в интернет.")
            self.label.config(text="Ошибка перевода (YC)", fg="#a80000")
        finally:
            current_status = self.label.cget("text")
            if "Перевод (YC) завершен" not in current_status and \
                    "Ошибка" not in current_status and \
                    "Тайм-аут" not in current_status and \
                    "соединения" not in current_status:
                if self.text_box.get("1.0", tk.END).strip():
                    self.label.config(text="Распознавание завершено", fg="#00695c")
                elif self.image_path:
                    self.label.config(text=os.path.basename(self.image_path), fg="#00695c")
                else:
                    self.label.config(text="Файл не выбран", fg="#a80000")

    def save_text(self):
        ocr_text = self.text_box.get("1.0", tk.END).strip()
        translated_text = ""
        original_state = self.translation_box.cget('state')
        self.translation_box.config(state=tk.NORMAL)
        translated_text = self.translation_box.get("1.0", tk.END).strip()
        self.translation_box.config(state=original_state)

        if not ocr_text and not translated_text:
            messagebox.showwarning("Внимание", "Нет текста для сохранения.")
            return

        text_to_save = ""
        if ocr_text:
            text_to_save += "Распознанный чувашский текст:\n" + ocr_text + "\n\n"
        if translated_text:
            text_to_save += "Перевод на русский (Yandex Cloud):\n" + translated_text + "\n"

        text_to_save = text_to_save.strip()

        file_path = filedialog.asksaveasfilename(
            title="Сохранить текст как...",
            defaultextension=".txt",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")])
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text_to_save)
                messagebox.showinfo("Успех", f"Текст успешно сохранён в:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить файл: {e}")

    def clear_all_fields(self):
        self._reset_ui_for_new_image()


if __name__ == "__main__":
    # Раскомментируйте и укажите свой путь, если Tesseract не в PATH
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    try:
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract OCR (версия {version}) найден.")
    except pytesseract.TesseractNotFoundError:
        print(
            "Tesseract OCR не найден! Укажите путь к tesseract.exe или добавьте его в PATH. Функциональность распознавания будет недоступна.")
        messagebox.showerror("Tesseract не найден",
                             "Tesseract OCR не найден. Пожалуйста, установите его и/или укажите путь к tesseract.exe в коде.")
    except Exception as e:
        print(f"Проблема с Tesseract: {e}")

    root = tk.Tk()
    root.geometry("800x850")
    root.minsize(600, 650)
    app = ChuvashOCRApp(root)

    root.mainloop()
