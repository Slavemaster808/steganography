import sys
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import hashlib
import json

class DigitalWatermarking:
    """
    Класс для внедрения и извлечения цифровых водяных знаков
    с использованием двух подходов: LSB и адаптивного по локальной дисперсии
    """
    
    def __init__(self, container_path):
        """
        Инициализация с изображением-контейнером
        
        Args:
            container_path: путь к изображению-контейнеру
        """
        try:
            # Загружаем изображение и конвертируем в grayscale
            self.container = Image.open(container_path).convert('L')
            self.container_array = np.array(self.container, dtype=np.uint8)
            self.height, self.width = self.container_array.shape
            self.total_pixels = self.height * self.width
            
            # Максимальная ёмкость контейнера (бит)
            self.max_capacity = self.total_pixels
            
            print(f"\n📦 Контейнер загружен: {self.width}x{self.height}")
            print(f"📊 Максимальная ёмкость: {self.max_capacity} бит ({self.max_capacity/8:.0f} байт)")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки контейнера: {e}")
            sys.exit(1)
    
    def prepare_watermark(self, watermark_path):
        """
        Подготовка ЦВЗ из изображения-логотипа
        
        Args:
            watermark_path: путь к изображению ЦВЗ
        
        Returns:
            биты ЦВЗ и исходное изображение для проверки
        """
        try:
            # Загружаем ЦВЗ и конвертируем в бинарное изображение
            watermark_img = Image.open(watermark_path).convert('L')
            
            # Сохраняем исходный размер
            original_size = (watermark_img.height, watermark_img.width)
            
            # Масштабируем ЦВЗ до размера, обеспечивающего 50% ёмкости контейнера
            target_bytes = self.max_capacity // 16  # 50% от ёмкости в байтах
            target_pixels = target_bytes * 8  # в битах
            
            # Вычисляем необходимые размеры ЦВЗ
            watermark_pixels = watermark_img.width * watermark_img.height
            
            if watermark_pixels > target_pixels:
                # Масштабируем вниз
                scale = (target_pixels / watermark_pixels) ** 0.5
                new_width = max(1, int(watermark_img.width * scale))
                new_height = max(1, int(watermark_img.height * scale))
                watermark_img = watermark_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print(f"  📐 Масштабирован до {new_width}x{new_height}")
            
            # Бинаризация (порог 128)
            watermark_array = np.array(watermark_img)
            binary_watermark = (watermark_array > 128).astype(np.uint8)
            
            # Преобразуем в одномерный массив битов
            watermark_bits = binary_watermark.flatten()
            
            # Достигаем 50% ёмкости путём повторения
            target_bits = self.max_capacity // 2
            if len(watermark_bits) < target_bits:
                repeats = (target_bits // len(watermark_bits)) + 1
                watermark_bits = np.tile(watermark_bits, repeats)
            
            watermark_bits = watermark_bits[:target_bits]
            
            print(f"  🔤 ЦВЗ подготовлен: {len(watermark_bits)} бит ({len(watermark_bits)/8:.0f} байт)")
            print(f"  📈 Доля от ёмкости: {len(watermark_bits)/self.max_capacity*100:.1f}%")
            
            return watermark_bits, watermark_img, original_size
            
        except Exception as e:
            print(f"❌ Ошибка подготовки ЦВЗ: {e}")
            return None, None, None
    
    def split_into_blocks(self, array, block_size=8):
        """
        Разделение изображения на блоки
        
        Args:
            array: входное изображение
            block_size: размер блока
        
        Returns:
            список блоков и метаданные
        """
        h, w = array.shape
        blocks = []
        
        # Вычисляем количество блоков
        n_blocks_h = h // block_size
        n_blocks_w = w // block_size
        
        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                block = array[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                blocks.append(block)
        
        metadata = {
            'block_size': block_size,
            'n_blocks_h': n_blocks_h,
            'n_blocks_w': n_blocks_w,
            'total_blocks': len(blocks),
            'original_shape': (h, w)
        }
        
        print(f"  📦 Изображение разделено на {len(blocks)} блоков {block_size}x{block_size}")
        return blocks, metadata
    
    def reconstruct_from_blocks(self, blocks, metadata):
        """
        Восстановление изображения из блоков
        
        Args:
            blocks: список блоков
            metadata: метаданные
        
        Returns:
            восстановленное изображение
        """
        block_size = metadata['block_size']
        n_blocks_h = metadata['n_blocks_h']
        n_blocks_w = metadata['n_blocks_w']
        
        h = n_blocks_h * block_size
        w = n_blocks_w * block_size
        
        reconstructed = np.zeros((h, w), dtype=np.uint8)
        
        block_idx = 0
        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                reconstructed[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = blocks[block_idx]
                block_idx += 1
        
        return reconstructed
    
    # ==================== РЕЖИМ 1: LSB ВНЕДРЕНИЕ ПО БЛОКАМ ====================
    
    def embed_lsb_blocks(self, watermark_bits, block_size=8, key_seed=None):
        """
        Внедрение ЦВЗ в LSB по блокам с использованием секретного ключа
        
        Args:
            watermark_bits: биты ЦВЗ
            block_size: размер блока
            key_seed: зерно для генерации ключа
        
        Returns:
            стего-изображение, метаданные
        """
        print("\n🔄 РЕЖИМ 1: Внедрение в LSB по блокам")
        
        # Разделяем изображение на блоки
        blocks, block_metadata = self.split_into_blocks(self.container_array, block_size)
        
        # Генерируем ключ для перемешивания блоков
        if key_seed is None:
            key_seed = random.randint(0, 2**32 - 1)
        
        random.seed(key_seed)
        block_indices = list(range(len(blocks)))
        random.shuffle(block_indices)
        
        # Подготавливаем биты ЦВЗ для внедрения
        bits_per_block = block_size * block_size  # по 1 биту на пиксель в LSB
        total_capacity = len(blocks) * bits_per_block
        
        if len(watermark_bits) > total_capacity:
            print(f"  ⚠️ ЦВЗ слишком большой, обрезаем до {total_capacity} бит")
            watermark_bits = watermark_bits[:total_capacity]
        
        # Внедряем биты в блоки
        modified_blocks = []
        bit_idx = 0
        
        for block_idx in block_indices:
            if bit_idx >= len(watermark_bits):
                modified_blocks.append(blocks[block_idx])
                continue
            
            block = blocks[block_idx].copy().flatten()
            bits_for_block = min(bits_per_block, len(watermark_bits) - bit_idx)
            
            for i in range(bits_for_block):
                block[i] = (block[i] & 0xFE) | watermark_bits[bit_idx + i]
            
            modified_blocks.append(block.reshape(block_size, block_size))
            bit_idx += bits_for_block
        
        # Восстанавливаем изображение
        stego_array = self.reconstruct_from_blocks(modified_blocks, block_metadata)
        
        # Метаданные для извлечения
        metadata = {
            'method': 'lsb_blocks',
            'block_size': block_size,
            'key_seed': key_seed,
            'embedded_bits': len(watermark_bits),
            'block_metadata': block_metadata,
            'block_indices': block_indices
        }
        
        print(f"  ✅ Внедрено {len(watermark_bits)} бит в {len(blocks)} блоков")
        print(f"  📍 Ключ: {key_seed}")
        
        return stego_array, metadata
    
    def extract_lsb_blocks(self, stego_path, metadata, original_size):
        """
        Извлечение ЦВЗ из LSB по блокам
        
        Args:
            stego_path: путь к стего-изображению
            metadata: метаданные внедрения
            original_size: исходный размер ЦВЗ
        
        Returns:
            извлечённые биты ЦВЗ и изображение
        """
        print("\n🔍 РЕЖИМ 1: Извлечение ЦВЗ из LSB по блокам")
        
        # Загружаем стего-изображение
        stego_img = Image.open(stego_path).convert('L')
        stego_array = np.array(stego_img)
        
        # Разделяем на блоки
        blocks, _ = self.split_into_blocks(stego_array, metadata['block_size'])
        
        # Восстанавливаем порядок блоков
        block_indices = metadata['block_indices']
        blocks_ordered = [blocks[i] for i in block_indices]
        
        # Извлекаем биты
        bits_per_block = metadata['block_size'] * metadata['block_size']
        embedded_bits = metadata['embedded_bits']
        
        extracted_bits = []
        bit_idx = 0
        
        for block in blocks_ordered:
            if bit_idx >= embedded_bits:
                break
            
            block_flat = block.flatten()
            bits_for_block = min(bits_per_block, embedded_bits - bit_idx)
            
            for i in range(bits_for_block):
                bit = block_flat[i] & 1
                extracted_bits.append(bit)
            
            bit_idx += bits_for_block
        
        extracted_bits = np.array(extracted_bits)
        print(f"  ✅ Извлечено {len(extracted_bits)} бит")
        
        # Восстанавливаем изображение ЦВЗ
        watermark_img = self.bits_to_image(extracted_bits, original_size)
        
        return extracted_bits, watermark_img
    
    # ==================== РЕЖИМ 2: АДАПТИВНОЕ ВНЕДРЕНИЕ ПО БЛОКАМ ====================
    
    def calculate_block_variance(self, block):
        """Вычисление дисперсии блока"""
        return np.var(block)
    
    def embed_adaptive_blocks(self, watermark_bits, block_size=8, variance_threshold=None):
        """
        Адаптивное внедрение по блокам на основе дисперсии
        
        В блоки с высокой дисперсией внедряем 2 бита на пиксель,
        с низкой дисперсией - 1 бит на пиксель
        
        Args:
            watermark_bits: биты ЦВЗ
            block_size: размер блока
            variance_threshold: порог дисперсии (если None - вычисляется автоматически)
        
        Returns:
            стего-изображение, метаданные
        """
        print("\n🔄 РЕЖИМ 2: Адаптивное внедрение по блокам")
        
        # Разделяем изображение на блоки
        blocks, block_metadata = self.split_into_blocks(self.container_array, block_size)
        
        # Вычисляем дисперсию для каждого блока
        block_variances = [self.calculate_block_variance(block) for block in blocks]
        
        # Определяем порог дисперсии (медиана или заданный)
        if variance_threshold is None:
            variance_threshold = np.median(block_variances)
        
        # Определяем битность для каждого блока
        block_bitness = []  # сколько бит на пиксель в блоке
        for var in block_variances:
            if var > variance_threshold:
                block_bitness.append(2)  # высокодисперсный блок - 2 бита
            else:
                block_bitness.append(1)  # низкодисперсный блок - 1 бит
        
        # Вычисляем общую ёмкость
        high_var_blocks = sum(1 for b in block_bitness if b == 2)
        low_var_blocks = len(blocks) - high_var_blocks
        bits_per_block_high = block_size * block_size * 2
        bits_per_block_low = block_size * block_size * 1
        total_capacity = high_var_blocks * bits_per_block_high + low_var_blocks * bits_per_block_low
        
        print(f"  📊 Высокодисперсных блоков: {high_var_blocks} (2 бита/пиксель)")
        print(f"  📊 Низкодисперсных блоков: {low_var_blocks} (1 бит/пиксель)")
        print(f"  📊 Общая ёмкость: {total_capacity} бит ({total_capacity/8:.0f} байт)")
        
        # Проверяем размер ЦВЗ
        if len(watermark_bits) > total_capacity:
            print(f"  ⚠️ ЦВЗ слишком большой, обрезаем до {total_capacity} бит")
            watermark_bits = watermark_bits[:total_capacity]
        
        # Внедряем биты в блоки
        modified_blocks = []
        bit_idx = 0
        
        for i, block in enumerate(blocks):
            if bit_idx >= len(watermark_bits):
                modified_blocks.append(block)
                continue
            
            block_flat = block.copy().flatten()
            bits_per_pixel = block_bitness[i]
            
            if bits_per_pixel == 1:
                # Внедряем 1 бит на пиксель
                pixels_for_block = min(len(block_flat), len(watermark_bits) - bit_idx)
                for j in range(pixels_for_block):
                    block_flat[j] = (block_flat[j] & 0xFE) | watermark_bits[bit_idx + j]
                bit_idx += pixels_for_block
            else:
                # Внедряем 2 бита на пиксель
                bits_needed = len(block_flat) * 2
                bits_available = len(watermark_bits) - bit_idx
                pixels_to_use = min(len(block_flat), bits_available // 2)
                
                for j in range(pixels_to_use):
                    if bit_idx + 1 < len(watermark_bits):
                        # Внедряем 2 бита
                        block_flat[j] = (block_flat[j] & 0xFC) | (watermark_bits[bit_idx] << 1) | watermark_bits[bit_idx + 1]
                        bit_idx += 2
                    else:
                        # Внедряем 1 бит если остался
                        block_flat[j] = (block_flat[j] & 0xFE) | watermark_bits[bit_idx]
                        bit_idx += 1
            
            modified_blocks.append(block_flat.reshape(block_size, block_size))
        
        # Восстанавливаем изображение
        stego_array = self.reconstruct_from_blocks(modified_blocks, block_metadata)
        
        # Метаданные для извлечения
        metadata = {
            'method': 'adaptive_blocks',
            'block_size': block_size,
            'variance_threshold': float(variance_threshold),
            'embedded_bits': len(watermark_bits),
            'block_metadata': block_metadata,
            'block_bitness': block_bitness,
            'block_variances': [float(v) for v in block_variances]
        }
        
        print(f"  ✅ Внедрено {len(watermark_bits)} бит")
        
        return stego_array, metadata
    
    def extract_adaptive_blocks(self, stego_path, metadata, original_size):
        """
        Извлечение ЦВЗ из адаптивно модифицированного изображения по блокам
        
        Args:
            stego_path: путь к стего-изображению
            metadata: метаданные внедрения
            original_size: исходный размер ЦВЗ
        
        Returns:
            извлечённые биты и восстановленное изображение
        """
        print("\n🔍 РЕЖИМ 2: Извлечение адаптивного ЦВЗ по блокам")
        
        # Загружаем стего-изображение
        stego_img = Image.open(stego_path).convert('L')
        stego_array = np.array(stego_img)
        
        # Разделяем на блоки
        blocks, _ = self.split_into_blocks(stego_array, metadata['block_size'])
        
        # Получаем битность блоков из метаданных
        block_bitness = metadata['block_bitness']
        embedded_bits = metadata['embedded_bits']
        
        # Извлекаем биты
        extracted_bits = []
        bit_idx = 0
        
        for i, block in enumerate(blocks):
            if bit_idx >= embedded_bits:
                break
            
            block_flat = block.flatten()
            bits_per_pixel = block_bitness[i]
            
            if bits_per_pixel == 1:
                # Извлекаем 1 бит на пиксель
                pixels_to_use = min(len(block_flat), embedded_bits - bit_idx)
                for j in range(pixels_to_use):
                    bit = block_flat[j] & 1
                    extracted_bits.append(bit)
                bit_idx += pixels_to_use
            else:
                # Извлекаем 2 бита на пиксель
                bits_needed = embedded_bits - bit_idx
                pixels_to_use = min(len(block_flat), (bits_needed + 1) // 2)
                
                for j in range(pixels_to_use):
                    if bit_idx + 1 < embedded_bits:
                        # Извлекаем 2 бита
                        bit1 = (block_flat[j] >> 1) & 1
                        bit2 = block_flat[j] & 1
                        extracted_bits.append(bit1)
                        extracted_bits.append(bit2)
                        bit_idx += 2
                    elif bit_idx < embedded_bits:
                        # Извлекаем 1 бит
                        bit = block_flat[j] & 1
                        extracted_bits.append(bit)
                        bit_idx += 1
        
        extracted_bits = np.array(extracted_bits[:embedded_bits])
        print(f"  ✅ Извлечено {len(extracted_bits)} бит")
        
        # Восстанавливаем изображение ЦВЗ
        watermark_img = self.bits_to_image(extracted_bits, original_size)
        
        return extracted_bits, watermark_img
    
    # ==================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ====================
    
    def bits_to_image(self, bits, original_size):
        """
        Преобразование битов обратно в изображение
        
        Args:
            bits: массив битов
            original_size: исходный размер (height, width)
        
        Returns:
            PIL Image
        """
        h, w = original_size
        total_pixels = h * w
        
        # Берём нужное количество бит
        if len(bits) < total_pixels:
            repeats = (total_pixels // len(bits)) + 1
            bits = np.tile(bits, repeats)
        
        bits = bits[:total_pixels]
        
        # Создаём изображение
        img_array = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(h):
            for j in range(w):
                idx = i * w + j
                img_array[i, j] = 255 if bits[idx] == 1 else 0
        
        return Image.fromarray(img_array)
    
    def calculate_psnr(self, original, stego):
        """
        Вычисление PSNR между оригиналом и стего-изображением
        
        Returns:
            PSNR в дБ
        """
        original = original.astype(np.float64)
        stego = stego.astype(np.float64)
        
        mse = np.mean((original - stego) ** 2)
        
        if mse == 0:
            return float('inf')
        
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        return psnr
    
    def create_comparison_image(self, original_wm, extracted_wm, output_path="comparison.png"):
        """
        Создание сравнительного изображения оригинал/извлечённый ЦВЗ
        """
        if original_wm is None or extracted_wm is None:
            return None
        
        # Приводим к одинаковому размеру
        if original_wm.size != extracted_wm.size:
            extracted_wm = extracted_wm.resize(original_wm.size, Image.Resampling.NEAREST)
        
        # Создаём полотно для сравнения
        width = original_wm.width * 3
        height = original_wm.height
        
        comparison = Image.new('L', (width, height), color=255)
        
        # Вставляем оригинал
        comparison.paste(original_wm, (0, 0))
        
        # Вставляем извлечённый
        comparison.paste(extracted_wm, (original_wm.width, 0))
        
        # Создаём изображение разницы
        orig_array = np.array(original_wm)
        ext_array = np.array(extracted_wm)
        diff_array = np.abs(orig_array.astype(np.int16) - ext_array.astype(np.int16))
        diff_array = (diff_array * 255).clip(0, 255).astype(np.uint8)
        diff_img = Image.fromarray(diff_array)
        
        comparison.paste(diff_img, (original_wm.width * 2, 0))
        
        # Добавляем подписи
        draw = ImageDraw.Draw(comparison)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 10), "Оригинал", fill=0, font=font)
        draw.text((original_wm.width + 10, 10), "Извлечено", fill=0, font=font)
        draw.text((original_wm.width * 2 + 10, 10), "Разница", fill=0, font=font)
        
        comparison.save(output_path)
        print(f"  🖼️ Сравнение сохранено: {output_path}")
        
        return comparison


def main():
    print("=" * 70)
    print("🔐 ПРОГРАММА ВНЕДРЕНИЯ ЦИФРОВЫХ ВОДЯНЫХ ЗНАКОВ ПО БЛОКАМ")
    print("=" * 70)
    
    # 1. Ввод пути к контейнеру
    while True:
        container_path = input("\n📁 Введите путь к изображению-контейнеру: ").strip()
        if os.path.exists(container_path):
            break
        print("❌ Файл не найден!")
    
    # Создаём объект
    dw = DigitalWatermarking(container_path)
    
    # 2. Ввод пути к ЦВЗ
    while True:
        watermark_path = input("\n🔤 Введите путь к изображению ЦВЗ (логотип): ").strip()
        if os.path.exists(watermark_path):
            break
        print("❌ Файл не найден!")
    
    # Подготавливаем ЦВЗ
    watermark_bits, original_watermark, original_size = dw.prepare_watermark(watermark_path)
    if watermark_bits is None:
        return
    
    # 3. Выбор режима
    print("\n" + "=" * 40)
    print("ВЫБЕРИТЕ РЕЖИМ ВНЕДРЕНИЯ:")
    print("1 - LSB внедрение по блокам")
    print("2 - Адаптивное внедрение по блокам")
    print("3 - Сравнить оба режима")
    print("=" * 40)
    
    mode = input("Ваш выбор (1/2/3): ").strip()
    
    # 4. Ввод размера блока
    block_size = 8
    try:
        block_size = int(input("\nВведите размер блока (по умолчанию 8): ").strip() or "8")
    except:
        block_size = 8
    
    results = {}
    
    # РЕЖИМ 1
    if mode in ['1', '3']:
        print("\n" + "-" * 60)
        print("РЕЖИМ 1: LSB внедрение по блокам")
        print("-" * 60)
        
        # Внедрение
        stego_lsb, lsb_metadata = dw.embed_lsb_blocks(watermark_bits, block_size)
        
        # Сохранение
        output_lsb = "stego_lsb_blocks.bmp"
        Image.fromarray(stego_lsb).save(output_lsb)
        
        # Сохраняем метаданные
        with open('lsb_blocks_metadata.json', 'w') as f:
            # Конвертируем для JSON
            metadata_serializable = {}
            for k, v in lsb_metadata.items():
                if isinstance(v, np.integer):
                    metadata_serializable[k] = int(v)
                elif isinstance(v, np.floating):
                    metadata_serializable[k] = float(v)
                elif isinstance(v, np.ndarray):
                    metadata_serializable[k] = v.tolist()
                elif isinstance(v, list):
                    metadata_serializable[k] = [int(x) if isinstance(x, np.integer) else x for x in v]
                else:
                    metadata_serializable[k] = v
            json.dump(metadata_serializable, f, indent=2)
        
        print(f"  💾 Стего: {output_lsb}")
        print(f"  💾 Метаданные: lsb_blocks_metadata.json")
        
        # PSNR
        psnr_lsb = dw.calculate_psnr(dw.container_array, stego_lsb)
        print(f"  📊 PSNR: {psnr_lsb:.2f} дБ")
        
        # Извлечение
        extracted_lsb_bits, extracted_lsb_img = dw.extract_lsb_blocks(
            output_lsb, lsb_metadata, original_size
        )
        
        # Сохраняем извлечённый ЦВЗ
        extracted_lsb_img.save("extracted_lsb_blocks.bmp")
        print(f"  💾 Извлечённый ЦВЗ: extracted_lsb_blocks.bmp")
        
        # Проверка совпадения
        match_len = min(len(extracted_lsb_bits), len(watermark_bits))
        match_lsb = np.sum(extracted_lsb_bits[:match_len] == watermark_bits[:match_len]) / match_len * 100
        print(f"  ✅ Совпадение битов: {match_lsb:.2f}%")
        
        # Создаём сравнение
        dw.create_comparison_image(original_watermark, extracted_lsb_img, "comparison_lsb_blocks.png")
        
        results['lsb'] = {'psnr': psnr_lsb, 'match': match_lsb}
    
    # РЕЖИМ 2
    if mode in ['2', '3']:
        print("\n" + "-" * 60)
        print("РЕЖИМ 2: Адаптивное внедрение по блокам")
        print("-" * 60)
        
        # Внедрение
        stego_adaptive, adaptive_metadata = dw.embed_adaptive_blocks(watermark_bits, block_size)
        
        # Сохранение
        output_adaptive = "stego_adaptive_blocks.bmp"
        Image.fromarray(stego_adaptive).save(output_adaptive)
        
        # Сохраняем метаданные
        with open('adaptive_blocks_metadata.json', 'w') as f:
            # Конвертируем для JSON
            metadata_serializable = {}
            for k, v in adaptive_metadata.items():
                if isinstance(v, np.integer):
                    metadata_serializable[k] = int(v)
                elif isinstance(v, np.floating):
                    metadata_serializable[k] = float(v)
                elif isinstance(v, np.ndarray):
                    metadata_serializable[k] = v.tolist()
                elif isinstance(v, list):
                    metadata_serializable[k] = [float(x) if isinstance(x, np.floating) else x for x in v]
                else:
                    metadata_serializable[k] = v
            json.dump(metadata_serializable, f, indent=2)
        
        print(f"  💾 Стего: {output_adaptive}")
        print(f"  💾 Метаданные: adaptive_blocks_metadata.json")
        
        # PSNR
        psnr_adaptive = dw.calculate_psnr(dw.container_array, stego_adaptive)
        print(f"  📊 PSNR: {psnr_adaptive:.2f} дБ")
        
        # Извлечение
        extracted_adaptive_bits, extracted_adaptive_img = dw.extract_adaptive_blocks(
            output_adaptive, adaptive_metadata, original_size
        )
        
        # Сохраняем извлечённый ЦВЗ
        extracted_adaptive_img.save("extracted_adaptive_blocks.bmp")
        print(f"  💾 Извлечённый ЦВЗ: extracted_adaptive_blocks.bmp")
        
        # Проверка совпадения
        match_len = min(len(extracted_adaptive_bits), len(watermark_bits))
        match_adaptive = np.sum(extracted_adaptive_bits[:match_len] == watermark_bits[:match_len]) / match_len * 100
        print(f"  ✅ Совпадение битов: {match_adaptive:.2f}%")
        
        # Создаём сравнение
        dw.create_comparison_image(original_watermark, extracted_adaptive_img, "comparison_adaptive_blocks.png")
        
        results['adaptive'] = {'psnr': psnr_adaptive, 'match': match_adaptive}
    
    # СРАВНЕНИЕ
    if mode == '3' and len(results) == 2:
        print("\n" + "=" * 60)
        print("📊 СРАВНЕНИЕ РЕЖИМОВ")
        print("=" * 60)
        print(f"LSB по блокам:        PSNR = {results['lsb']['psnr']:.2f} дБ, Совпадение = {results['lsb']['match']:.2f}%")
        print(f"Адаптивный по блокам: PSNR = {results['adaptive']['psnr']:.2f} дБ, Совпадение = {results['adaptive']['match']:.2f}%")
    
    print("\n✅ Программа завершена!")
    print("📁 Результаты сохранены в текущей директории")


def test_with_sample():
    """
    Тестовая функция для демонстрации работы
    """
    print("\n🧪 ЗАПУСК ТЕСТОВОЙ ДЕМОНСТРАЦИИ\n")
    
    # Создаём тестовый контейнер
    test_container = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            test_container[i, j] = (i + j) % 256
    Image.fromarray(test_container).save("test_container.bmp")
    
    # Создаём тестовый ЦВЗ
    watermark = np.zeros((64, 64), dtype=np.uint8)
    # Рисуем букву "Ц"
    watermark[20:44, 28:36] = 255  # вертикальная линия
    watermark[20:28, 28:44] = 255  # верхняя горизонталь
    watermark[36:44, 28:44] = 255  # нижняя горизонталь
    
    Image.fromarray(watermark).save("test_watermark.bmp")
    
    print("📁 Созданы тестовые файлы:")
    print("   - test_container.bmp (256x256)")
    print("   - test_watermark.bmp (64x64)")
    
    # Запускаем демонстрацию
    dw = DigitalWatermarking("test_container.bmp")
    bits, orig_wm, orig_size = dw.prepare_watermark("test_watermark.bmp")
    
    # Тест LSB по блокам
    print("\n" + "=" * 60)
    stego_lsb, lsb_meta = dw.embed_lsb_blocks(bits, 16)
    psnr_lsb = dw.calculate_psnr(dw.container_array, stego_lsb)
    print(f"PSNR (LSB blocks): {psnr_lsb:.2f} дБ")
    
    extracted_lsb, _ = dw.extract_lsb_blocks("test_container.bmp", lsb_meta, orig_size)
    match_lsb = np.sum(extracted_lsb[:len(bits)] == bits) / len(bits) * 100
    print(f"Совпадение LSB: {match_lsb:.2f}%")
    
    # Тест адаптивного по блокам
    print("\n" + "=" * 60)
    stego_adapt, adapt_meta = dw.embed_adaptive_blocks(bits, 16)
    psnr_adapt = dw.calculate_psnr(dw.container_array, stego_adapt)
    print(f"PSNR (Adaptive blocks): {psnr_adapt:.2f} дБ")
    
    extracted_adapt, _ = dw.extract_adaptive_blocks("test_container.bmp", adapt_meta, orig_size)
    match_adapt = np.sum(extracted_adapt[:len(bits)] == bits) / len(bits) * 100
    print(f"Совпадение Adaptive: {match_adapt:.2f}%")
    
    print("\n✅ Тест завершён!")


if __name__ == "__main__":
    # Для теста:
    # test_with_sample()
    
    # Для основной программы:
    main()