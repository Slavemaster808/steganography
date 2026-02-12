import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import hashlib
import sys


class SimpleBitPlaneProcessor:
    """Упрощенный процессор битовых плоскостей для верификации"""
    
    def __init__(self, image_path):
        """
        Инициализация процессора битовых плоскостей
        
        Args:
            image_path: путь к исходному изображению
        """
        try:
            self.original_image = Image.open(image_path).convert('L')
            self.image_array = np.array(self.original_image, dtype=np.uint8)
            self.height, self.width = self.image_array.shape
            self.total_pixels = self.height * self.width
            self.image_path = image_path
            print(f"✓ Изображение загружено: {self.width}x{self.height}, {self.total_pixels} пикселей")
        except Exception as e:
            print(f"✗ Ошибка загрузки изображения: {e}")
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
        
        bit_position = k - 1
        bit_mask = np.uint8(1 << bit_position)
        
        bit_plane = (self.image_array & bit_mask) >> bit_position
        binary_image = (bit_plane * 255).astype(np.uint8)
        
        return binary_image
    
    def save_bit_plane(self, k, output_path):
        """Сохраняет k-ю битовую плоскость как изображение"""
        bit_plane = self.extract_bit_plane(k)
        img = Image.fromarray(bit_plane)
        img.save(output_path)
        print(f"Битовая плоскость {k} сохранена в {output_path}")


class SimpleBitPlaneVerifier:
    """Упрощенный верификатор для проверки внедрения"""
    
    def __init__(self, original_path, embedded_path):
        """
        Инициализация верификатора
        
        Args:
            original_path: путь к исходному изображению
            embedded_path: путь к изображению с внедренным сообщением
        """
        self.original_path = original_path
        self.embedded_path = embedded_path
        
        try:
            self.original = np.array(Image.open(original_path).convert('L'), dtype=np.uint8)
            self.embedded = np.array(Image.open(embedded_path).convert('L'), dtype=np.uint8)
            self.height, self.width = self.original.shape
            
            print(f"✓ Изображения загружены для сравнения")
        except Exception as e:
            print(f"✗ Ошибка загрузки изображений: {e}")
            sys.exit(1)
    
    def create_comparison_image(self, k=None, output_path="comparison.png"):
        """Создает визуальное сравнение с помощью PIL"""
        
        # Создаем полотно для сравнения
        comparison = Image.new('L', (self.width * 3, self.height * (2 if k else 1)), color=255)
        
        # Оригинал
        original_img = Image.fromarray(self.original)
        comparison.paste(original_img, (0, 0))
        
        # Модифицированное
        embedded_img = Image.fromarray(self.embedded)
        comparison.paste(embedded_img, (self.width, 0))
        
        # Разница
        difference = np.abs(self.original.astype(np.int16) - self.embedded.astype(np.int16))
        diff_normalized = (difference * 3).clip(0, 255).astype(np.uint8)  # Увеличиваем контраст
        diff_img = Image.fromarray(diff_normalized)
        comparison.paste(diff_img, (self.width * 2, 0))
        
        if k:
            bit_pos = k - 1
            mask = 1 << bit_pos
            
            # Оригинал битовая плоскость
            orig_bit = ((self.original & mask) >> bit_pos) * 255
            orig_bit_img = Image.fromarray(orig_bit.astype(np.uint8))
            comparison.paste(orig_bit_img, (0, self.height))
            
            # Модифицированная битовая плоскость
            emb_bit = ((self.embedded & mask) >> bit_pos) * 255
            emb_bit_img = Image.fromarray(emb_bit.astype(np.uint8))
            comparison.paste(emb_bit_img, (self.width, self.height))
            
            # Разница в битовой плоскости
            bit_diff = (orig_bit != emb_bit).astype(np.uint8) * 255
            bit_diff_img = Image.fromarray(bit_diff)
            comparison.paste(bit_diff_img, (self.width * 2, self.height))
        
        # Добавляем подписи
        draw = ImageDraw.Draw(comparison)
        try:
            # Пробуем разные пути для шрифтов
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "/Library/Fonts/Arial.ttf",
                "C:\\Windows\\Fonts\\Arial.ttf"
            ]
            font = None
            for path in font_paths:
                if os.path.exists(path):
                    font = ImageFont.truetype(path, 16)
                    break
            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 10), "Оригинал", fill=0, font=font)
        draw.text((self.width + 10, 10), "С сообщением", fill=0, font=font)
        draw.text((self.width * 2 + 10, 10), "Разница (x3)", fill=0, font=font)
        
        if k:
            draw.text((10, self.height + 10), f"Бит {k} (ориг)", fill=0, font=font)
            draw.text((self.width + 10, self.height + 10), f"Бит {k} (с сообщ.)", fill=0, font=font)
            draw.text((self.width * 2 + 10, self.height + 10), "Измененные биты", fill=0, font=font)
        
        comparison.save(output_path)
        print(f"✓ Сравнительное изображение сохранено: {output_path}")
        return output_path
    
    def print_statistics(self):
        """Выводит статистику изменений"""
        diff = np.abs(self.original.astype(np.int16) - self.embedded.astype(np.int16))
        changed_pixels = np.sum(diff > 0)
        total_pixels = self.original.size
        
        print("\n" + "=" * 60)
        print("СТАТИСТИКА ИЗМЕНЕНИЙ")
        print("=" * 60)
        print(f"Всего пикселей: {total_pixels:,}")
        print(f"Изменено пикселей: {changed_pixels:,} ({changed_pixels/total_pixels*100:.4f}%)")
        print(f"Среднее изменение: {np.mean(diff):.4f}")
        print(f"Максимальное изменение: {np.max(diff)}")
        print(f"Минимальное изменение: {np.min(diff)}")
        print("=" * 60)
    
    def verify_bit_plane_integrity(self, k):
        """Проверяет, сколько бит было изменено в конкретной плоскости"""
        bit_pos = k - 1
        mask = 1 << bit_pos
        
        orig_bits = (self.original >> bit_pos) & 1
        emb_bits = (self.embedded >> bit_pos) & 1
        
        changed_bits = np.sum(orig_bits != emb_bits)
        total_bits = self.original.size
        
        print(f"\nАнализ битовой плоскости {k}:")
        print(f"   Изменено бит: {changed_bits:,} из {total_bits:,} ({changed_bits/total_bits*100:.4f}%)")
        
        return changed_bits


class SimpleMessageExtractor:
    """Упрощенный экстрактор сообщений"""
    
    def __init__(self, image_path):
        """Инициализация экстрактора"""
        self.image_path = image_path
        try:
            self.image = np.array(Image.open(image_path).convert('L'), dtype=np.uint8)
            self.height, self.width = self.image.shape
            print(f"✓ Изображение загружено для извлечения: {self.width}x{self.height}")
        except Exception as e:
            print(f"✗ Ошибка загрузки изображения: {e}")
            sys.exit(1)
    
    def extract_from_bitplane(self, k, num_bits=None):
        """Извлечение сообщения из битовой плоскости"""
        bit_position = k - 1
        flat_array = self.image.flatten()
        
        if num_bits is None:
            num_bits = len(flat_array)
        
        num_bits = min(num_bits, len(flat_array))
        
        extracted_bits = []
        for i in range(num_bits):
            bit = (flat_array[i] >> bit_position) & 1
            extracted_bits.append(str(bit))
        
        # Конвертируем биты в байты
        bytes_data = bytearray()
        bit_string = ''.join(extracted_bits)
        
        for i in range(0, len(bit_string), 8):
            if i + 8 <= len(bit_string):
                byte = int(bit_string[i:i+8], 2)
                bytes_data.append(byte)
        
        return bytes(bytes_data)
    
    def save_extracted_message(self, k, output_file, num_bits=None):
        """Сохраняет извлеченное сообщение"""
        print(f"\nИзвлечение сообщения из битовой плоскости {k}...")
        
        extracted_data = self.extract_from_bitplane(k, num_bits)
        
        with open(output_file, 'wb') as f:
            f.write(extracted_data)
        
        print(f"✓ Сообщение извлечено!")
        print(f"  Размер: {len(extracted_data):,} байт ({len(extracted_data)/1024:.2f} KB)")
        print(f"  Сохранено в: {output_file}")
        
        # Показываем первые байты
        print("\n📋 Первые 64 байта извлеченного сообщения:")
        print("-" * 70)
        
        for i in range(0, min(64, len(extracted_data)), 16):
            # Hex представление
            hex_part = ' '.join(f'{b:02x}' for b in extracted_data[i:i+16])
            # ASCII представление
            ascii_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in extracted_data[i:i+16])
            print(f"{i:04x}: {hex_part:<48} {ascii_part}")
        
        return extracted_data
    
    def compare_with_original(self, k, original_file):
        """Сравнивает извлеченное сообщение с оригиналом"""
        try:
            with open(original_file, 'rb') as f:
                original_data = f.read()
        except Exception as e:
            print(f"✗ Ошибка чтения оригинального файла: {e}")
            return False
        
        extracted_data = self.extract_from_bitplane(k, len(original_data) * 8)
        extracted_data = extracted_data[:len(original_data)]
        
        print("\n" + "=" * 70)
        print("ПРОВЕРКА ЦЕЛОСТНОСТИ СООБЩЕНИЯ")
        print("=" * 70)
        
        # MD5 хеши
        original_hash = hashlib.md5(original_data).hexdigest()
        extracted_hash = hashlib.md5(extracted_data).hexdigest()
        
        print(f"📝 MD5 оригинального: {original_hash}")
        print(f"📝 MD5 извлеченного:  {extracted_hash}")
        
        if original_hash == extracted_hash:
            print("\nУСПЕХ: Сообщение полностью восстановлено без ошибок!")
            return True
        else:
            print("\nОШИБКА: Сообщения не совпадают!")
            
            # Находим первое несовпадение
            for i in range(min(len(original_data), len(extracted_data))):
                if original_data[i] != extracted_data[i]:
                    print(f"\n   Первое несовпадение на байте {i}:")
                    print(f"     Оригинал: 0x{original_data[i]:02x} ({original_data[i]:3d}) '{chr(original_data[i]) if 32 <= original_data[i] < 127 else '.'}'")
                    print(f"     Извлечено: 0x{extracted_data[i]:02x} ({extracted_data[i]:3d}) '{chr(extracted_data[i]) if 32 <= extracted_data[i] < 127 else '.'}'")
                    break
            
            return False


def verify_embedding():
    """Основная функция проверки внедрения"""
    
    print("\n" + "=" * 70)
    print("ПРОВЕРКА ВНЕДРЕНИЯ СООБЩЕНИЯ В БИТОВУЮ ПЛОСКОСТЬ")
    print("=" * 70)
    
    # Получаем пути к файлам
    print("\n📁 Введите пути к файлам:")
    original = input("   Исходное изображение: ").strip()
    embedded = input("   Изображение с сообщением: ").strip()
    
    if not os.path.exists(original):
        print(f"✗ Ошибка: Файл '{original}' не найден!")
        return
    
    if not os.path.exists(embedded):
        print(f"✗ Ошибка: Файл '{embedded}' не найден!")
        return
    
    try:
        k = int(input("   Номер битовой плоскости для анализа (1-8): ").strip())
        if k < 1 or k > 8:
            print("✗ Ошибка: Номер бита должен быть от 1 до 8!")
            return
    except ValueError:
        print("✗ Ошибка: Введите число от 1 до 8!")
        return
    
    # 1. Создаем верификатор
    print("\n🔧 Инициализация верификатора...")
    verifier = SimpleBitPlaneVerifier(original, embedded)
    
    # 2. Визуальное сравнение
    print("\nСоздание визуального сравнения...")
    output_comparison = f"comparison_bit{k}.png"
    verifier.create_comparison_image(k, output_comparison)
    
    # 3. Статистика
    verifier.print_statistics()
    verifier.verify_bit_plane_integrity(k)
    
    # 4. Извлечение сообщения
    print("\nИзвлечение сообщения...")
    extractor = SimpleMessageExtractor(embedded)
    output_message = f"extracted_bit{k}.bin"
    extractor.save_extracted_message(k, output_message)
    
    # 5. Опциональное сравнение с оригиналом
    print("\n" + "-" * 70)
    compare = input("Хотите сравнить с оригинальным файлом сообщения? (y/n): ").lower()
    if compare == 'y':
        original_msg = input("   Путь к оригинальному файлу сообщения: ").strip()
        if os.path.exists(original_msg):
            extractor.compare_with_original(k, original_msg)
        else:
            print(f"✗ Файл '{original_msg}' не найден!")
    
    print("\n" + "=" * 70)
    print("ПРОВЕРКА ЗАВЕРШЕНА!")
    print("=" * 70)
    print(f"Отчет сохранен: {output_comparison}")
    print(f"Сообщение извлечено: {output_message}")


def extract_message_only():
    """Функция только для извлечения сообщения"""
    
    print("\n" + "=" * 70)
    print("🔍 ИЗВЛЕЧЕНИЕ СООБЩЕНИЯ ИЗ БИТОВОЙ ПЛОСКОСТИ")
    print("=" * 70)
    
    img_path = input("\n📁 Путь к изображению с сообщением: ").strip()
    
    if not os.path.exists(img_path):
        print(f"✗ Ошибка: Файл '{img_path}' не найден!")
        return
    
    try:
        k = int(input("   Номер битовой плоскости (1-8): ").strip())
        if k < 1 or k > 8:
            print("✗ Ошибка: Номер бита должен быть от 1 до 8!")
            return
    except ValueError:
        print("✗ Ошибка: Введите число от 1 до 8!")
        return
    
    output = input("   Файл для сохранения сообщения: ").strip()
    if not output:
        output = f"extracted_bit{k}.bin"
    
    try:
        max_bits = input("   Максимум бит для извлечения (Enter - все): ").strip()
        num_bits = int(max_bits) if max_bits else None
    except ValueError:
        num_bits = None
    
    extractor = SimpleMessageExtractor(img_path)
    extractor.save_extracted_message(k, output, num_bits)
    
    print(f"\n✅ Сообщение извлечено и сохранено в '{output}'")


def demo_mode():
    """Демонстрационный режим"""
    
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИОННЫЙ РЕЖИМ")
    print("=" * 70)
    
    print("\n1. Создание тестовых файлов...")
    
    # Создаем тестовое изображение
    test_image = np.random.randint(100, 200, (200, 200), dtype=np.uint8)
    Image.fromarray(test_image).save("demo_original.png")
    print("   ✓ Создано: demo_original.png (200x200, случайное изображение)")
    
    # Создаем тестовое сообщение
    test_message = "🔐 Это тестовое сообщение для проверки стеганографии! " * 50
    with open("demo_message.txt", "w", encoding='utf-8') as f:
        f.write(test_message)
    print(f"   ✓ Создано: demo_message.txt ({len(test_message)} байт)")
    
    print("\n2. Для внедрения сообщения используйте основную программу")
    print("   или вставьте этот код в ваш процессор.")
    
    print("\n3. Для проверки внедрения запустите режим 1")
    print("   и укажите созданные файлы:")
    print("   - Исходное: demo_original.png")
    print("   - С сообщением: [ваш файл после внедрения]")
    
    print("\n4. Для извлечения сообщения используйте режим 3")
    
    print("\n Демонстрационные файлы созданы!")


def main():
    """Главное меню программы"""
    
    while True:
        print("\n" + "=" * 70)
        print("ВЕРИФИКАТОР БИТОВЫХ ПЛОСКОСТЕЙ")
        print("=" * 70)
        print("1. Проверить внедрение сообщения")
        print("2. Демонстрационный режим")
        print("3. Извлечь сообщение из изображения")
        print("4. Извлечь битовую плоскость")
        print("5. Выход")
        print("=" * 70)
        
        choice = input("Выберите режим (1-5): ").strip()
        
        if choice == '1':
            verify_embedding()
        elif choice == '2':
            demo_mode()
        elif choice == '3':
            extract_message_only()
        elif choice == '4':
            # Извлечение битовой плоскости
            img_path = input("\n📁 Путь к изображению: ").strip()
            if os.path.exists(img_path):
                try:
                    k = int(input("   Номер битовой плоскости (1-8): "))
                    output = input("   Файл для сохранения: ").strip()
                    if not output:
                        output = f"bitplane_{k}.png"
                    
                    processor = SimpleBitPlaneProcessor(img_path)
                    processor.save_bit_plane(k, output)
                except ValueError as e:
                    print(f"✗ Ошибка: {e}")
            else:
                print(f"✗ Файл '{img_path}' не найден!")
        elif choice == '5':
            print("\n👋 Программа завершена. До свидания!")
            break
        else:
            print("✗ Неверный выбор. Пожалуйста, выберите 1-5.")


if __name__ == "__main__":
    main()