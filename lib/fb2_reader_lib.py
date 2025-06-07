import chardet                              # Для автоопределение кодировки
import sys      # Для sys.exit()
import os       # Для работы с файловой системой  
import re
from bs4 import BeautifulSoup

from constants_lib import write_to_a_text_file
from constants_lib import print_a_tag_map


def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def process_filename(filename):
    filename = filename.strip().replace("  ", " ")
    filename = filename.replace(" ", "_")
    #filename = re.sub(r'\s', '_', filename)
    # Удаление недопустимых символов
    filename = re.sub(r'[^\w\-_.]', '', filename)
    return filename

# --- Блок чтения файла и подготовки данных
def read_and_preprocessing_files(folder_path):
    print("Чтение файлов из папки ", folder_path, ":")
    fb2_files = [file for file in os.listdir(folder_path) if file.endswith('.fb2')] # Получаем список файлов с расширением .fb2 в указанной папке
    fb2_files_sorted = sorted(fb2_files)
    soups = []
    soups.append(read_and_preprocessing_file(0, folder_path, fb2_files_sorted))
    soups.append(read_and_preprocessing_file(1, folder_path, fb2_files_sorted))
    return soups

def read_and_preprocessing_file(number, folder_path, fb2_files):
    fb2_file_path = os.path.join(folder_path, fb2_files[number])        # Формируем полный путь к исходному файлу
    soup = read_file_to_soup(fb2_file_path)
    print(fb2_files[number]," - прочитан как Файл ", number, ".")
    
    body = get_body(soup, number)
        
    if write_to_a_text_file:
        write_txt_file(body, fb2_file_path)
    return body
    
def read_file_to_soup(fb2_file_path):
    encoding = detect_encoding(fb2_file_path)                           # Автоопределение кодировки
    with open(fb2_file_path, 'r', encoding=encoding) as file:           # Открываем исходный файл и парсим его содержимое с помощью BeautifulSoup
        soup = BeautifulSoup(file, 'xml')
    return soup

def get_body(soup, number):
    bodies = soup.find_all('body')                                        # Находим тег <body>
    if len(bodies) == 0:
        print(f'\nОшибка в файле {number}. Тег <body> не найден. Для дальнейшей работы будет использован полный текст.\n')
        return soup
    elif len(bodies) == 1:
        return bodies[0]
    else:
        print(f'\nПредупреждение! В файле {number} обнаружено несколько тегов <body>. Следующиий текст не будет обработан:')
        for body in bodies[1:]:
            print(body.get_text()[:500])
        #print('\n')
        return bodies[0]

def write_txt_file(body, fb2_file_path):
    text = body.get_text()                                              # Получаем текст из тега <body>
    txt_file_path = os.path.splitext(fb2_file_path)[0] + '.txt'         # Формируем путь к целевому файлу .txt с тем же названием
    with open(txt_file_path, 'w', encoding='utf-8') as file:            # Записываем содержимое в целевой файл, перезаписывая его при необходимости
        file.write(text)
# --- Блок чтения файла и подготовки данных


# --- Блок проверки структуры
def initial_structure_check(soups):
    n_sections_0 = fb2_structure_check(0, soups)
    n_sections_1 = fb2_structure_check(1, soups)
    if n_sections_0 != n_sections_1:
        print(f'\nПредупреждение! Количество глав не совпадает!\n В файле 0 {n_sections_0} глав.\n В файле 1 {n_sections_1} глав.')
    return min(n_sections_0, n_sections_1)

def fb2_structure_check(number, soups):
    n_sections = tag_counting(number, soups)
    
    if print_a_tag_map:
        tag_maping(number, soups)

    return n_sections

def tag_counting(number, soups):
    n_sections = 0
    top_level_tags = [tag for tag in soups[number].find_all(recursive=False)]
    for tag in top_level_tags:
        if tag.name == 'title':
            None
        elif tag.name == 'section':
            n_sections += 1
        elif tag.name == 'empty-line':
            None
        elif tag.name == 'image':
            None
        else:
            print(f'\nПредупреждение! В файле {number} обнаружен тег {tag.name} на верхнем уровне. Следующиий текст не будет обработан:')
            print(tag.get_text()[:500])
    print( "Файл ", number, " содержит ", n_sections,"section." )
    return n_sections

def tag_maping(number, soups):
    collected_tags = ""
    old_tag = ""
    sections = soups[number].find_all('section')
    for section in sections:
        top_level_tags = [tag.name for tag in section.find_all(recursive=False)]
        for tag in top_level_tags:
            if tag != 'empty-line':
                if old_tag != 'p' or tag != 'p':
                    collected_tags += tag + " "
                old_tag = tag
        collected_tags += "\n"
    print( collected_tags )
# --- Блок проверки структуры




