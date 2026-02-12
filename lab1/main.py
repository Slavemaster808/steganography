import sys
import os
from PIL import Image
import numpy as np

class BitPlaneProcessor:
    def __init__(self, image_path):
        """
        Инициализация процессора битовых плоскостей
        
        Args:
            image_path: путь к исходному изображению
        """
        try:
            # Загружаем изображение и конвертируем в grayscale
            self.original_image = Image.open(image_path).convert('L')
            self.image_array = np.array(self.original_image, dtype=np.uint8)
            self.height, self.width = self.image_array.shape
            self.total_pixels = self.height * self.width
            print(f"Изображение загружено. Размер: {self.width}x{self.height}")
            print(f"Всего пикселей: {self.total_pixels}")
        except Exception as e:
            print(f"Ошибка загрузки изображения: {e}")
            sys.exit(1)
    
    def extract_bit_plane(self, k):
        """
        Извлечение k-й битовой плоскости
        
        Args:
            k: номер бита (1-8, где 1 - младший бит)
        
        Returns:
            бинарное изображение как numpy array
        """
        if k < 1 or k > 8:
            raise ValueError("Номер бита должен быть от 1 до 8")
        
        # Создаем маску для k-го бита (сдвиг на k-1 позиций)
        bit_position = k - 1
        bit_mask = np.uint8(1 << bit_position)
        
        # Извлекаем биты
        bit_plane = (self.image_array & bit_mask) >> bit_position
        
        # Преобразуем в черно-белое (0 и 255)
        binary_image = (bit_plane * 255).astype(np.uint8)
        
        return binary_image
    
    def save_bit_plane(self, k, output_path):
        """
        Сохраняет k-ю битовую плоскость как изображение
        """
        bit_plane = self.extract_bit_plane(k)
        img = Image.fromarray(bit_plane)
        img.save(output_path)
        print(f"Битовая плоскость {k} сохранена в {output_path}")
    
    def embed_message(self, k, message_file, output_path):
        """
        Внедрение сообщения в k-ю битовую плоскость
        
        Args:
            k: номер бита для замены (1-8)
            message_file: путь к файлу с сообщением
            output_path: путь для сохранения результата
        """
        if k < 1 or k > 8:
            raise ValueError("Номер бита должен быть от 1 до 8")
        
        bit_position = k - 1
        bit_mask = np.uint8(1 << bit_position)
        clear_mask = np.uint8(~bit_mask)  # Маска для очистки бита
        
        try:
            # Читаем сообщение из файла
            with open(message_file, 'rb') as f:
                message_bytes = f.read()
            
            # Проверяем размер сообщения
            if len(message_bytes) < 30 * 1024:  # 30KB
                print(f"Предупреждение: Размер сообщения ({len(message_bytes)} байт) меньше 30КБ")
            
            # Конвертируем байты в биты
            message_bits = []
            for byte in message_bytes:
                for bit_pos in range(7, -1, -1):  # Старший бит первым
                    bit = (byte >> bit_pos) & 1
                    message_bits.append(bit)
            
            max_bits_to_write = self.total_pixels
            bits_to_write = min(len(message_bits), max_bits_to_write)
            
            # Создаем копию изображения для внедрения
            modified_array = self.image_array.copy()
            
            # Уплощаем массив для последовательной обработки
            flat_array = modified_array.flatten()
            
            # Внедряем биты
            written_bits = 0
            for i in range(bits_to_write):
                # Очищаем k-й бит
                flat_array[i] = flat_array[i] & clear_mask
                # Устанавливаем новое значение бита
                flat_array[i] = flat_array[i] | np.uint8(message_bits[i] << bit_position)
                written_bits += 1
            
            # Возвращаем исходную форму
            modified_array = flat_array.reshape(self.height, self.width)
            
            # Убеждаемся, что значения в допустимом диапазоне
            modified_array = np.clip(modified_array, 0, 255).astype(np.uint8)
            
            # Сохраняем результат
            result_image = Image.fromarray(modified_array)
            result_image.save(output_path)
            
            print(f"Успешно внедрено {written_bits} бит ({written_bits / 8:.1f} байт)")
            print(f"Максимально возможная емкость: {max_bits_to_write} бит ({max_bits_to_write / 8} байт)")
            print(f"Результат сохранен в {output_path}")
            
            return written_bits
            
        except FileNotFoundError:
            print(f"Ошибка: Файл сообщения '{message_file}' не найден")
        except Exception as e:
            print(f"Ошибка при внедрении сообщения: {e}")
            import traceback
            traceback.print_exc()

    def embed_message_safe(self, k, message_file, output_path):
        """
        Альтернативный, более безопасный метод внедрения сообщения
        """
        if k < 1 or k > 8:
            raise ValueError("Номер бита должен быть от 1 до 8")
        
        bit_position = k - 1
        
        try:
            # Читаем сообщение из файла
            with open(message_file, 'rb') as f:
                message_bytes = f.read()
            
            # Проверяем размер сообщения
            if len(message_bytes) < 30 * 1024:
                print(f"Предупреждение: Размер сообщения ({len(message_bytes)} байт) меньше 30КБ")
            
            # Конвертируем байты в биты
            message_bits = []
            for byte in message_bytes:
                for bit_pos in range(7, -1, -1):
                    bit = (byte >> bit_pos) & 1
                    message_bits.append(bit)
            
            max_bits_to_write = self.total_pixels
            bits_to_write = min(len(message_bits), max_bits_to_write)
            
            # Создаем копию изображения
            modified_array = self.image_array.copy().astype(np.int16)
            
            # Уплощаем массив
            flat_array = modified_array.flatten()
            
            # Внедряем биты
            written_bits = 0
            for i in range(bits_to_write):
                # Получаем текущее значение пикселя
                pixel_value = int(flat_array[i])
                
                # Очищаем нужный бит
                pixel_value = pixel_value & ~(1 << bit_position)
                
                # Устанавливаем новый бит
                pixel_value = pixel_value | (message_bits[i] << bit_position)
                
                # Записываем обратно
                flat_array[i] = pixel_value
                written_bits += 1
            
            # Возвращаем форму и конвертируем обратно в uint8
            modified_array = flat_array.reshape(self.height, self.width)
            modified_array = np.clip(modified_array, 0, 255).astype(np.uint8)
            
            # Сохраняем результат
            result_image = Image.fromarray(modified_array)
            result_image.save(output_path)
            
            print(f"Успешно внедрено {written_bits} бит ({written_bits / 8:.1f} байт)")
            print(f"Результат сохранен в {output_path}")
            
            return written_bits
            
        except Exception as e:
            print(f"Ошибка при внедрении сообщения: {e}")
            import traceback
            traceback.print_exc()

def main():
    # print("=" * 60)
    # print("ПРОГРАММА ДЛЯ РАБОТЫ С БИТОВЫМИ ПЛОСКОСТЯМИ")
    # print("=" * 60)
    
    # Запрашиваем путь к изображению
    while True:
        image_path = input("\nВведите путь к исходному изображению: ").strip()
        if os.path.exists(image_path):
            break
        print("Файл не найден. Попробуйте снова.")
    
    # Создаем процессор
    processor = BitPlaneProcessor(image_path)
    
    while True:
        print("\n" + "=" * 40)
        print("ВЫБЕРИТЕ РЕЖИМ РАБОТЫ:")
        print("1 - Извлечение битовой плоскости")
        print("2 - Внедрение сообщения в битовую плоскость")
        print("3 - Выход")
        print("=" * 40)
        
        mode = input("Ваш выбор (1/2/3): ").strip()
        
        if mode == '1':
            # Режим извлечения битовой плоскости
            try:
                k = int(input("Введите номер бита для извлечения (1-8): "))
                output_path = input("Введите путь для сохранения результата: ").strip()
                
                # Добавляем расширение если его нет
                if not output_path.lower().endswith(('.png', '.jpg', '.bmp', '.tiff')):
                    output_path += '.png'
                
                processor.save_bit_plane(k, output_path)
                
            except ValueError as e:
                print(f"Ошибка: {e}")
            except Exception as e:
                print(f"Произошла ошибка: {e}")
        
        elif mode == '2':
            # Режим внедрения сообщения
            try:
                k = int(input("Введите номер бита для внедрения (1-8): "))
                message_path = input("Введите путь к файлу с сообщением: ").strip()
                
                if not os.path.exists(message_path):
                    print("Файл с сообщением не найден!")
                    continue
                
                output_path = input("Введите путь для сохранения результата: ").strip()
                
                # Добавляем расширение если его нет
                if not output_path.lower().endswith(('.png', '.jpg', '.bmp', '.tiff')):
                    output_path += '.png'
                
                print("\nВыберите метод внедрения:")
                print("1 - Стандартный")
                print("2 - Безопасный (рекомендуется)")
                method = input("Ваш выбор (1/2): ").strip()
                
                if method == '2':
                    processor.embed_message_safe(k, message_path, output_path)
                else:
                    processor.embed_message(k, message_path, output_path)
                
            except ValueError as e:
                print(f"Ошибка: {e}")
            except Exception as e:
                print(f"Произошла ошибка: {e}")
        
        elif mode == '3':
            print("Программа завершена.")
            break
        
        else:
            print("Неверный выбор. Пожалуйста, выберите 1, 2 или 3.")

def test_with_sample():
    """
    Тестовая функция для проверки работы
    """
    print("\n=== ТЕСТИРОВАНИЕ ПРОГРАММЫ ===\n")
    
    # Создаем тестовое изображение 10x10
    test_data = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
    test_img = Image.fromarray(test_data)
    test_img.save("test_input.png")
    
    # Создаем тестовое сообщение
    with open("test_message.txt", "w") as f:
        f.write("Hello, World! This is a test message." * 1000)
    
    # Тестируем
    processor = BitPlaneProcessor("test_input.png")
    
    # Тест извлечения
    print("\n--- Тест извлечения битовой плоскости ---")
    for k in [1, 4, 8]:
        processor.save_bit_plane(k, f"test_bitplane_{k}.png")
    
    # Тест внедрения
    print("\n--- Тест внедрения сообщения ---")
    processor.embed_message_safe(1, "test_message.txt", "test_output.png")
    
    print("\nТестирование завершено!")

if __name__ == "__main__":
    # Для тестирования раскомментируйте следующую строку:
    # test_with_sample()
    
    # Запуск основной программы
    main()