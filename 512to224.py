from PIL import Image
import os

# Указываем путь к папке с изображениями
input_folder = '/home/emil/Рабочий стол/Дипл/Data/train_masks'
output_folder = '/home/emil/Рабочий стол/Дипл/Data/train224_masks'

# Создаем директорию для сохраненных изображений, если ее нет
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Проходимся по всем файлам в папке
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)
    
    try:
        # Открываем изображение
        img = Image.open(file_path)
        
        # Проверяем, что изображение действительно 512x512
        if img.size == (512, 512):
            # Изменяем размер изображения до 224x224
            resized_img = img.resize((224, 224), resample=Image.BICUBIC)
            
            # Сохраняем новое изображение в указанную папку
            output_file_path = os.path.join(output_folder, filename)
            resized_img.save(output_file_path)
            
            print(f"Изображение {filename} успешно изменено.")
        else:
            print(f"Изображение {filename} не 512x512, пропускаем...")
    except OSError as e:
        print(f"Произошла ошибка при обработке файла {file_path}: {e}")
