import requests
import os
from PIL import Image
from io import BytesIO


cookies = {
    '_ga': 'GA1.1.1591888550.1693809521',
    '_ga_R1FN4KJKJH': 'GS1.1.1693812648.2.0.1693812648.0.0.0',
}

headers = {
    'Accept': '*/*',
    'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
    'Connection': 'keep-alive',
    'Content-Type': 'multipart/form-data; boundary=----WebKitFormBoundaryvAwa86HKk1xhiDI0',
    # 'Cookie': '_ga=GA1.1.1591888550.1693809521; _ga_R1FN4KJKJH=GS1.1.1693812648.2.0.1693812648.0.0.0',
    'Origin': 'http://127.0.0.1:56946',
    'Referer': 'http://127.0.0.1:56946/',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'User-Agent': 'Mozilla/5.0 (Linux; Android 10; SM-G981B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.162 Mobile Safari/537.36',
    'sec-ch-ua': '"Not_A Brand";v="99", "Google Chrome";v="109", "Chromium";v="109"',
    'sec-ch-ua-mobile': '?1',
    'sec-ch-ua-platform': '"Android"',
    'pipka': 'lox'
}

def get_absolute_image_paths(directory_path):
    image_paths = []

    # Проверяем, существует ли директория
    if not os.path.exists(directory_path):
        print(f"Директория '{directory_path}' не существует.")
        return image_paths

    # Получаем список файлов в директории
    files = os.listdir(directory_path)

    # Цикл для обработки каждого файла
    for file_name in files:
        file_path = file_name

        # Проверяем, является ли файл изображением
        if file_name.endswith((".jpg", ".jpeg", ".png", ".gif")):
            # Добавляем абсолютный путь к изображению в список
            image_paths.append(file_path)
            print(f"Найдено изображение: {file_path}")

    return image_paths

image_paths =get_absolute_image_paths('/workspaces/lama-cleaner/images')

for index, image in enumerate(image_paths):
    with open(f'images/{image}', 'rb') as image_file:


        # Создаем словарь с данными для отправки, включая изображение


        # Отправляем POST-запрос с изображением на сервер
        # response = requests.post(url, files=files)
        response = requests.post('http://0.0.0.0:8080/in', 
        cookies=cookies, 
        headers=headers,
        data=image_file)

        if response.status_code == 200:
            print(f'[OK] {index} save image with name: ', image)
            img = Image.open(BytesIO(response.content))
            img.save(f'output/{image}')
        else:
            print('[ERROR] ',response.status_code, 'sorry')