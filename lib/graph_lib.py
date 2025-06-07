import os
import numpy as np
import matplotlib.pyplot as plt

from constants_lib import image_folder
from constants_lib import goal_tokens_limit


def clear_image_folder():
    folder_path = os.path.join(os.getcwd(), image_folder)
    # Удаляем все файлы в папке
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)




def draw_graph(data, name):
    file_path = os.path.join(os.getcwd(), image_folder, name)
    
    x = [item[0] for item in data]
    y = [item[1] for item in data]
    # Полиномиальная аппроксимация
    coefficients = np.polyfit(x, y, 1)
    polynomial = np.poly1d(coefficients)
    trend_line = polynomial(x)
    # Строим график
    plt.plot(x, y, marker='o', label='Данные')
    plt.plot(x, trend_line, label='Линия тренда')
    # Называем оси и даем заголовок графика
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('График с линией тренда, ' + name)
    # Добавляем легенду
    plt.legend()
    plt.ylim(0, goal_tokens_limit)  # Установка пределов значений по оси y
    # Отображаем график
    #plt.show()
    plt.savefig(file_path, dpi=150)  # Сохранение в файл
    plt.clf()


def draw_nubmer_list(data, number_of_graph):
    #print(data)
    list_for_draw = []
    name = f'graph_{number_of_graph}.png'
    for i, number in enumerate(data):
        list_for_draw.append([i, number])
    draw_graph(list_for_draw, name)
