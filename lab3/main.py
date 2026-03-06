import struct
import os
import math
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass
import shutil


class BMPHeader:
    """Структура заголовка BMP файла"""
    def __init__(self):
        self.bfType = 0x4D42  # 'BM'
        self.bfSize = 0
        self.bfReserved1 = 0
        self.bfReserved2 = 0
        self.bfOffBits = 0
        self.biSize = 40  # Размер информационного заголовка
        self.biWidth = 0
        self.biHeight = 0
        self.biPlanes = 1
        self.biBitCount = 8  # 8 бит на пиксель (градации серого)
        self.biCompression = 0
        self.biSizeImage = 0
        self.biXPelsPerMeter = 0
        self.biYPelsPerMeter = 0
        self.biClrUsed = 0
        self.biClrImportant = 0

    def pack(self) -> bytes:
        """Упаковывает заголовок в байты"""
        # Формат: H=uint16, I=uint32, i=int32
        return struct.pack('<HIHHIIiiHHIIIIII',
                          self.bfType, self.bfSize, self.bfReserved1, self.bfReserved2,
                          self.bfOffBits, self.biSize, self.biWidth, self.biHeight,
                          self.biPlanes, self.biBitCount, self.biCompression,
                          self.biSizeImage, self.biXPelsPerMeter, self.biYPelsPerMeter,
                          self.biClrUsed, self.biClrImportant)

    def unpack(self, data: bytes):
        """Распаковывает заголовок из байтов"""
        (self.bfType, self.bfSize, self.bfReserved1, self.bfReserved2,
         self.bfOffBits, self.biSize, self.biWidth, self.biHeight,
         self.biPlanes, self.biBitCount, self.biCompression,
         self.biSizeImage, self.biXPelsPerMeter, self.biYPelsPerMeter,
         self.biClrUsed, self.biClrImportant) = struct.unpack('<HIHHIIiiHHIIIIII', data[:54])


class GrayBMP:
    """Класс для работы с BMP изображениями в градациях серого"""
    
    def __init__(self):
        self.header = BMPHeader()
        self.palette = bytearray(1024)  # 256 цветов * 4 байта (BGRX)
        self.pixels = bytearray()
        self.width = 0
        self.height = 0
        self.loaded = False
        
        # Инициализация стандартной grayscale палитры
        for i in range(256):
            self.palette[i*4:i*4+4] = bytes([i, i, i, 0])

    def load(self, filename: str) -> bool:
        """Загружает BMP файл"""
        try:
            with open(filename, 'rb') as file:
                # Читаем заголовок
                header_data = file.read(54)
                if len(header_data) < 54:
                    return False
                    
                self.header.unpack(header_data)
                
                if self.header.bfType != 0x4D42 or self.header.biBitCount != 8:
                    return False
                
                self.width = self.header.biWidth
                self.height = abs(self.header.biHeight)
                
                # Читаем палитру
                file.seek(54, os.SEEK_SET)
                self.palette = bytearray(file.read(1024))
                
                # Читаем данные пикселей
                file.seek(self.header.bfOffBits, os.SEEK_SET)
                row_size = ((self.width * 8 + 31) // 32) * 4
                data_size = row_size * self.height
                raw_data = file.read(data_size)
                
                # Преобразуем в построчный формат (без выравнивания)
                self.pixels = bytearray(self.width * self.height)
                for y in range(self.height):
                    src_y = (self.height - 1 - y) if self.header.biHeight > 0 else y
                    row_start = src_y * row_size
                    for x in range(self.width):
                        self.pixels[y * self.width + x] = raw_data[row_start + x]
                
                self.loaded = True
                return True
                
        except Exception as e:
            print(f"Error loading BMP: {e}")
            return False

    def save(self, filename: str) -> bool:
        """Сохраняет BMP файл"""
        if not self.loaded:
            return False
            
        try:
            row_size = ((self.width * 8 + 31) // 32) * 4
            data_size = row_size * self.height
            
            # Подготавливаем данные с выравниванием строк
            raw_data = bytearray(data_size)
            for y in range(self.height):
                dst_y = (self.height - 1 - y) if self.header.biHeight > 0 else y
                row_start = dst_y * row_size
                for x in range(self.width):
                    raw_data[row_start + x] = self.pixels[y * self.width + x]
            
            # Обновляем заголовок
            self.header.bfOffBits = 54 + 1024
            self.header.bfSize = self.header.bfOffBits + data_size
            self.header.biSizeImage = data_size
            
            with open(filename, 'wb') as file:
                file.write(self.header.pack())
                file.write(self.palette)
                file.write(raw_data)
            
            return True
            
        except Exception as e:
            print(f"Error saving BMP: {e}")
            return False

    def get_width(self) -> int:
        return self.width

    def get_height(self) -> int:
        return self.height

    def get_size(self) -> int:
        return self.width * self.height

    def data(self) -> bytearray:
        return self.pixels

    def get_pixels(self) -> bytearray:
        return self.pixels.copy()

    def set_pixels(self, new_pixels: bytearray):
        self.pixels = new_pixels

    def clone(self):
        copy = GrayBMP()
        copy.header = self.header
        copy.palette = self.palette.copy()
        copy.width = self.width
        copy.height = self.height
        copy.pixels = self.pixels.copy()
        copy.loaded = self.loaded
        return copy


class Metrics:
    """Класс для вычисления метрик качества"""
    
    @staticmethod
    def compute_psnr(original: GrayBMP, stego: GrayBMP) -> float:
        """Вычисляет PSNR между двумя изображениями"""
        size = original.get_width() * original.get_height()
        orig_pixels = original.data()
        stego_pixels = stego.data()
        
        mse = 0.0
        for i in range(size):
            diff = orig_pixels[i] - stego_pixels[i]
            mse += diff * diff
        
        mse /= size
        
        if mse == 0:
            return float('inf')
        
        return 10 * math.log10((255 * 255) / mse)


@dataclass
class PeakZeroPair:
    """Структура для хранения пары пик-ноль"""
    peak: int
    zero: int
    peak_count: int = 0


class HistogramShiftingEmbedder:
    """Класс для встраивания данных методом сдвига гистограммы"""
    
    def __init__(self):
        self.pairs = []

    def read_data_from_file(self, filename: str) -> bytes:
        """Читает данные из файла"""
        try:
            with open(filename, 'rb') as file:
                return file.read()
        except Exception as e:
            print(f"Error: Cannot open file {filename}")
            return b''

    def write_data_to_file(self, data: bytes, filename: str) -> bool:
        """Записывает данные в файл"""
        try:
            with open(filename, 'wb') as file:
                file.write(data)
            return True
        except Exception as e:
            print(f"Error: Cannot create file {filename}")
            return False

    def _compute_histogram(self, image: GrayBMP) -> Dict[int, int]:
        """Вычисляет гистограмму изображения"""
        hist = {i: 0 for i in range(256)}
        pixels = image.data()
        size = image.get_width() * image.get_height()
        
        for i in range(size):
            hist[pixels[i]] += 1
        
        return hist

    def _find_peak_zero_pairs(self, hist: Dict[int, int], required_capacity: int) -> List[PeakZeroPair]:
        """Находит пары пик-ноль в гистограмме"""
        pairs = []
        
        # Находим нулевые точки
        zero_points = [i for i in range(256) if hist[i] == 0]
        
        total_capacity = 0
        last_zero = -1
        
        for zero in zero_points:
            if total_capacity >= required_capacity:
                break
            
            if last_zero + 1 < zero:
                # Находим пик между последним нулем и текущим нулем
                peak = last_zero + 1
                max_count = hist[peak]
                
                for i in range(last_zero + 2, zero):
                    if hist[i] > max_count:
                        max_count = hist[i]
                        peak = i
                
                if max_count > 0:
                    pairs.append(PeakZeroPair(peak=peak, zero=zero, peak_count=max_count))
                    total_capacity += max_count
            
            last_zero = zero
        
        return pairs

    def embed(self, container: GrayBMP, data: bytes, stego: GrayBMP, 
              metadata: Dict[str, int]) -> bool:
        """Встраивает данные в изображение"""
        
        required_capacity = len(data) * 8
        hist = self._compute_histogram(container)
        total_pixels = container.get_width() * container.get_height()
        
        if required_capacity > total_pixels:
            print(f"Error: Data too large. Required: {required_capacity} bits, Available: {total_pixels} bits")
            return False
        
        self.pairs = self._find_peak_zero_pairs(hist, required_capacity)
        if not self.pairs:
            print("Not enough capacity! Could not find suitable peak-zero pairs.")
            return False
        
        # Копируем изображение
        stego.__dict__.update(container.clone().__dict__)
        pixels = stego.data()
        size = container.get_width() * container.get_height()
        
        # Сортируем пары по убыванию количества пикселей в пике
        self.pairs.sort(key=lambda x: x.peak_count, reverse=True)
        
        # Сохраняем метаданные
        metadata["num_pairs"] = len(self.pairs)
        metadata["data_size"] = len(data)
        for i, pair in enumerate(self.pairs):
            metadata[f"peak_{i}"] = pair.peak
            metadata[f"zero_{i}"] = pair.zero
        
        # Встраиваем данные
        bit_index = 0
        total_bits = len(data) * 8
        
        for pair in self.pairs:
            if bit_index >= total_bits:
                break
            
            peak = pair.peak
            zero = pair.zero
            shift_right = zero > peak
            
            # Первый проход: сдвигаем пиксели
            for i in range(size):
                if shift_right:
                    if peak < pixels[i] < zero:
                        pixels[i] += 1
                else:
                    if zero < pixels[i] < peak:
                        pixels[i] -= 1
            
            # Второй проход: встраиваем биты в пиковые значения
            for i in range(size):
                if bit_index >= total_bits:
                    break
                    
                if pixels[i] == peak:
                    byte_index = bit_index // 8
                    bit_position = 7 - (bit_index % 8)
                    bit = (data[byte_index] >> bit_position) & 1
                    
                    if bit == 1:
                        if shift_right:
                            pixels[i] += 1
                        else:
                            pixels[i] -= 1
                    
                    bit_index += 1
        
        return True

    def extract(self, stego: GrayBMP, metadata: Dict[str, int], 
                extracted_data: List[bytes], restored: GrayBMP) -> bool:
        """Извлекает данные из изображения"""
        
        # Копируем изображение
        restored.__dict__.update(stego.clone().__dict__)
        pixels = restored.data()
        size = restored.get_width() * restored.get_height()
        
        num_pairs = metadata["num_pairs"]
        data_size = metadata["data_size"]
        
        pairs = []
        for i in range(num_pairs):
            peak = metadata[f"peak_{i}"]
            zero = metadata[f"zero_{i}"]
            pairs.append(PeakZeroPair(peak=peak, zero=zero))
        
        bits = []
        
        # Извлекаем данные в обратном порядке пар
        for pair in reversed(pairs):
            peak = pair.peak
            zero = pair.zero
            shift_right = zero > peak
            
            # Извлекаем биты
            extracted_bits = []
            for i in range(size):
                if shift_right:
                    if pixels[i] == peak + 1:
                        extracted_bits.append(1)
                        pixels[i] = peak
                    elif pixels[i] == peak:
                        extracted_bits.append(0)
                else:
                    if pixels[i] == peak - 1:
                        extracted_bits.append(1)
                        pixels[i] = peak
                    elif pixels[i] == peak:
                        extracted_bits.append(0)
            
            bits = extracted_bits + bits
            
            # Восстанавливаем сдвинутые пиксели
            for i in range(size):
                if shift_right:
                    if peak < pixels[i] <= zero:
                        pixels[i] -= 1
                else:
                    if zero <= pixels[i] < peak:
                        pixels[i] += 1
        
        # Преобразуем биты в байты
        extracted_bytes = bytearray()
        for i in range(data_size):
            byte = 0
            for b in range(8):
                bit_pos = i * 8 + b
                if bit_pos < len(bits) and bits[bit_pos]:
                    byte |= (1 << (7 - b))
            extracted_bytes.append(byte)
        
        extracted_data.clear()
        extracted_data.append(bytes(extracted_bytes))
        
        return True

    def save_metadata(self, metadata: Dict[str, int], filename: str):
        """Сохраняет метаданные в файл"""
        with open(filename, 'w') as file:
            for key, value in metadata.items():
                file.write(f"{key} {value}\n")

    def load_metadata(self, filename: str) -> Dict[str, int]:
        """Загружает метаданные из файла"""
        metadata = {}
        try:
            with open(filename, 'r') as file:
                for line in file:
                    key, value = line.strip().split()
                    metadata[key] = int(value)
        except Exception as e:
            print(f"Error loading metadata: {e}")
        return metadata


def verify_data(original: bytes, extracted: List[bytes]) -> bool:
    """Проверяет соответствие исходных и извлеченных данных"""
    if not extracted:
        return False
    return original == extracted[0]


def test_dataset(dataset_path: str, dataset_name: str, 
                 data_file_path: str, output_dir: str):
    """Тестирует метод на наборе изображений"""
    
    print(f"\n{'='*10} Testing on {dataset_name} dataset {'='*10}")
    
    # Создаем необходимые директории
    os.makedirs(f"{output_dir}/{dataset_name}/stego", exist_ok=True)
    os.makedirs(f"{output_dir}/{dataset_name}/restored", exist_ok=True)
    os.makedirs(f"{output_dir}/{dataset_name}/extracted", exist_ok=True)
    os.makedirs(f"{output_dir}/{dataset_name}/metadata", exist_ok=True)
    
    embedder = HistogramShiftingEmbedder()
    total_psnr = 0.0
    success_count = 0
    total_images = 0
    psnr_values = []
    
    # Читаем данные для встраивания
    test_data = embedder.read_data_from_file(data_file_path)
    if not test_data:
        print(f"Failed to read data file: {data_file_path}")
        return
    
    print(f"Data file: {data_file_path} ({len(test_data)} bytes)")
    
    # Открываем файл для записи результатов
    with open(f"{output_dir}/{dataset_name}_results.txt", 'w') as results_file:
        results_file.write(f"Dataset: {dataset_name}\n")
        results_file.write(f"Data file: {data_file_path} ({len(test_data)} bytes)\n")
        results_file.write("="*40 + "\n\n")
        
        # Обрабатываем все BMP файлы в директории
        for entry in Path(dataset_path).iterdir():
            if entry.suffix.lower() != '.bmp':
                continue
            
            total_images += 1
            filename = entry.stem
            print(f"\n[{total_images}] Processing: {filename}.bmp")
            
            # Загружаем изображение
            container = GrayBMP()
            if not container.load(str(entry)):
                print(f"  Failed to load image")
                results_file.write(f"{filename}.bmp: FAILED (cannot load)\n")
                continue
            
            required_bits = len(test_data) * 8
            total_pixels = container.get_width() * container.get_height()
            
            if required_bits > total_pixels:
                print(f"  Skipping - image too small")
                results_file.write(f"{filename}.bmp: SKIPPED (too small)\n")
                continue
            
            # Встраиваем данные
            stego = GrayBMP()
            metadata = {}
            
            if not embedder.embed(container, test_data, stego, metadata):
                print(f"  Embedding failed")
                results_file.write(f"{filename}.bmp: FAILED (embedding)\n")
                continue
            
            # Вычисляем PSNR
            psnr = Metrics.compute_psnr(container, stego)
            psnr_values.append(psnr)
            total_psnr += psnr
            success_count += 1
            
            print(f"  PSNR = {psnr:.2f} dB")
            results_file.write(f"{filename}.bmp: PSNR = {psnr:.2f} dB\n")
            
            # Сохраняем стего-изображение
            stego_path = f"{output_dir}/{dataset_name}/stego/{filename}_stego.bmp"
            stego.save(stego_path)
            
            # Сохраняем метаданные
            metadata_path = f"{output_dir}/{dataset_name}/metadata/{filename}_metadata.txt"
            embedder.save_metadata(metadata, metadata_path)
            
            # Извлекаем данные
            restored = GrayBMP()
            extracted_data = []
            
            if not embedder.extract(stego, metadata, extracted_data, restored):
                print(f"  Extraction failed")
                results_file.write(f"  Extraction: FAILED\n")
                continue
            
            # Сохраняем восстановленное изображение
            restored_path = f"{output_dir}/{dataset_name}/restored/{filename}_restored.bmp"
            restored.save(restored_path)
            
            # Сохраняем извлеченные данные
            extracted_path = f"{output_dir}/{dataset_name}/extracted/{filename}_extracted.txt"
            if embedder.write_data_to_file(extracted_data[0], extracted_path):
                print(f"  Extracted data saved to: {extracted_path}")
            
            # Проверяем данные
            if verify_data(test_data, extracted_data):
                print(f"  Data successfully verified")
                results_file.write(f"  Extraction: SUCCESS (data matches)\n")
            else:
                print(f"  Data verification failed")
                results_file.write(f"  Extraction: FAILED (data mismatch)\n")
            
            # Проверяем восстановление изображения
            restore_psnr = Metrics.compute_psnr(container, restored)
            if restore_psnr > 99.0:
                print(f"  Image perfectly restored")
            else:
                print(f"  Image restoration error: {restore_psnr:.2f} dB")
        
        # Записываем итоговую статистику
        results_file.write("\n" + "="*40 + "\n")
        results_file.write(f"Total images processed: {total_images}\n")
        results_file.write(f"Successfully embedded: {success_count}\n")
        
        if success_count > 0:
            avg_psnr = total_psnr / success_count
            results_file.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
            
            if len(psnr_values) > 1:
                variance = sum((p - avg_psnr) ** 2 for p in psnr_values) / (len(psnr_values) - 1)
                std_dev = math.sqrt(variance)
                t = 1.96  # для 95% доверительного интервала
                
                results_file.write(f"Std deviation: {std_dev:.2f} dB\n")
                ci_lower = avg_psnr - t * std_dev / math.sqrt(len(psnr_values))
                ci_upper = avg_psnr + t * std_dev / math.sqrt(len(psnr_values))
                results_file.write(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}] dB\n")
            
            print(f"\n--- Results for {dataset_name} ---")
            print(f"Successfully processed: {success_count}/{total_images} images")
            print(f"Average PSNR: {avg_psnr:.2f} dB")


def main():
    # Пути к директориям с изображениями
    boss_path = "../lab1/set1"
    medical_path = "../lab1/set2"
    flowers_path = "../lab1/set3"
    
    data_file_path = "message.txt"
    output_dir = "results"
    
    # Создаем выходную директорию
    os.makedirs(output_dir, exist_ok=True)
    
    # Создаем пример файла с сообщением, если его нет
    if not os.path.exists(data_file_path):
        print("\nFile message.txt not found. Creating sample file...")
        with open(data_file_path, 'w', encoding='utf-8') as sample_file:
            sample_file.write("This is a sample secret message for steganography testing.\n")
            sample_file.write("The quick brown fox jumps over the lazy dog.\n")
            sample_file.write("0123456789 !@#$%^&*()_+{}[]|\\:;\"'<>,.?/~\n")
            sample_file.write("Histogram shifting is a reversible data hiding technique.\n")
            sample_file.write("Multiple pairs of peak and zero points are used for embedding.\n")
        print("Sample message.txt created.")
    
    # Тестируем на каждом наборе
    test_dataset(boss_path, "BOSS", data_file_path, output_dir)
    test_dataset(medical_path, "Medical", data_file_path, output_dir)
    test_dataset(flowers_path, "Flowers", data_file_path, output_dir)
    
    # Создаем сводный файл с результатами
    with open(f"{output_dir}/summary.txt", 'w') as summary:
        summary.write("="*40 + "\n")
        summary.write("HISTOGRAM SHIFTING STEGANOGRAPHY RESULTS\n")
        summary.write("="*40 + "\n\n")
        summary.write(f"Data file: {data_file_path}\n\n")
        
        for dataset in ["BOSS", "Medical", "Flowers"]:
            results_path = f"{output_dir}/{dataset}_results.txt"
            if os.path.exists(results_path):
                with open(results_path, 'r') as results:
                    summary.write(results.read())
                    summary.write("\n")
        
        summary.write("="*40 + "\n")
    
    print(f"\n{'='*40}")
    print("Testing complete!")
    print(f"Results saved in: {output_dir}/")
    print(f"Summary: {output_dir}/summary.txt")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()