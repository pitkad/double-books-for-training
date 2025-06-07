from scipy.spatial.distance import cosine
from razdel import sentenize
from tokenizers import Tokenizer
import numpy as np
from sentence_transformers import SentenceTransformer       # Импортируем библиотеку для векторизации текста
import sys      # Для sys.exit()

import graph_lib
import inspect
from constants_lib import show_progress_bar
from constants_lib import batch_of_vectors
from constants_lib import seq_length
from constants_lib import print_token_overflow
from constants_lib import limit_of_max_outlier
from constants_lib import goal_tokens_limit

from knapsack_solver_lib import solver


model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')   # Определяем модель для векторизации текста

tokenizer_st = Tokenizer.from_file("tokenizer-sentence-transformers_distiluse-base-multilingual-cased-v2.json")
tokenizer_goal = Tokenizer.from_file("tokenizer-mistralai_Mistral-7B-Instruct-v0.2.json")


# --- Блок получения списка предложений главы с векторами и параметрами
def create_sentences_list(section):
    sentences =[]
    paragraphs = section.find_all(recursive=False)
    for paragraph in paragraphs:
        processed_text = [paragraph.get_text()]
        sentences_from_paragraph = text_preprocessing(processed_text, paragraph.name)
        sentences.extend(sentences_from_paragraph)
    sentences = vectorization_and_tokenization(sentences)
    return sentences

def text_preprocessing(processed_text, paragraph_name):             # Загрузка текста книг, анализ и формирование
    sentences_from_paragraph = []
    for elem in processed_text:
        senten = [sentence.text for sentence in sentenize(elem)] # Разбиваем абзацы на предложения
        # Структура: предложение[0], вектор[1], конец абзаца[2], токенов ST[3], токенов целевой модели[4], соответствие [5], обработан? [6], статус [7], вид абазаца [8]
        sentences_from_paragraph.extend([[senten[i], 0, i == len(senten)-1, 0, 0, 0, False, "НЕ ОБРАБ", paragraph_name] for i in range(len(senten))])
    return sentences_from_paragraph
    
def vectorization_and_tokenization(sentences):             # Загрузка текста книг, анализ и формирование
    sentences_only = [sublist[0] for sublist in sentences]
    embeddings = model.encode(sentences_only, batch_size=10, normalize_embeddings=True, show_progress_bar=show_progress_bar)  # Получаем векторы для каждого предложения   False True
    for i, element in enumerate(sentences):
        element[1] = embeddings[i]
        element[3] = len(tokenizer_st.encode(element[0]))
        element[4] = len(tokenizer_goal.encode(element[0]))
        if element[3] > 128 and print_token_overflow:
            print("Переполнение токенов:", element[3], ". В строке: ", element[0])
    return sentences
# --- Блок получения списка предложений главы с векторами и параметрами



def первоначальный_поиск_соответствий(sentences_1, sentences_2):
    start_index_1 = 0
    start_index_2 = 0
    completed = False
    while True:
        current_batch_of_vectors = batch_of_vectors         # Сбрасываем длину пакета
        while True:
            next_element_1 = find_closest_vector(sentences_1, sentences_2, start_index_1, start_index_2, current_batch_of_vectors, seq_length) 
            next_element_2 = find_closest_vector(sentences_2, sentences_1, start_index_2, start_index_1, current_batch_of_vectors, seq_length) 
            # Если последним предложениям найдена пара - сразу идем на выход
            if sentences_1[len(sentences_1)-1][5] != 0 and sentences_2[len(sentences_2)-1][5] != 0:                     
                completed = True
                break                               # Выход из цикла
            if next_element_1 is not None and next_element_2 is not None:               # Прежде всего проверяем, что не попалось None
                if next_element_1 == sentences_2[next_element_2][5] and next_element_2 == sentences_1[next_element_1][5]:   # Предложения совпали, идем на обработку следующей партии
                    start_index_1 = next_element_1
                    start_index_2 = next_element_2
                    break
                elif next_element_1 < sentences_2[next_element_2][5] and sentences_1[next_element_1][5] < next_element_2 :
                    new_next_element = process_sequence(sentences_2, start_index_2, sentences_1[next_element_1][5]+seq_length+1, 0)
                    if next_element_1 == sentences_2[new_next_element][5] and new_next_element == sentences_1[next_element_1][5] and new_next_element != 0:
                        start_index_1 = next_element_1
                        start_index_2 = new_next_element
                        break
                elif next_element_1 > sentences_2[next_element_2][5] and sentences_1[next_element_1][5] > next_element_2 :
                    new_next_element = process_sequence(sentences_1, start_index_1, sentences_2[next_element_2][5]+seq_length+1, 0)
                    if new_next_element == sentences_2[next_element_2][5] and next_element_2 == sentences_1[new_next_element][5] and new_next_element != 0:
                        start_index_1 = new_next_element
                        start_index_2 = next_element_2
                        break
            current_batch_of_vectors += 1
        if completed:
            break                               # Выход из цикла
    return sentences_1, sentences_2

def find_closest_vector(sentences_1, sentences_2, start_index_1, start_index_2, batch_of_vectors, seq_length):
    vectors_i = [value[1] for value in sentences_1[start_index_1:start_index_1+batch_of_vectors]]       # Извлечение векторов из списка sentences
    vectors_j = [value[1] for value in sentences_2[start_index_2:start_index_2+batch_of_vectors]]       # Извлечение векторов из списка sentences
    for i, vector in enumerate(vectors_i):      # Обход первого массива векторов
        min_distance = float('inf')
        for j, vec in enumerate(vectors_j):
            distance = cosine(vector, vec)
            if distance < min_distance:
                min_distance = distance
                closest_index = j
        sentences_1[i+start_index_1][5] = closest_index+start_index_2
    return process_sequence(sentences_1, start_index_1, start_index_1+batch_of_vectors, None)

def process_sequence(sentences, start_index, end_index, value_if_invalid):
    sequence = [value[5] for value in sentences[start_index:end_index]]
    sequence_result = find_sequence(sequence, seq_length)
    if sequence_result is None or sequence_result == 0:
        new_next_element = value_if_invalid
    else:
        new_next_element = sequence_result + start_index
    return new_next_element

def find_sequence(numbers, seq_length):                     # Поиск последнего значения в списке, после которого идет заданное количество элементов, каждый из который увеличивается на 1
    for i in range(len(numbers) - seq_length - 1, -1, -1):
        sequence_found = True                               # Значение True сохранится если для данного элемента выполнены условия
        for j in range(seq_length):                         # Проверяем каждый элемент после текущего числа в пределах заданного seq_length
            if numbers[i] + j + 1 != numbers[i + j + 1]:    # Проверяем условие последовательного увеличения чисел
                sequence_found = False
                break
        if sequence_found:                                  # Если для элемента условия выполнены, возвращаем индекс текущего числа
            return i
    return None                                             # Если не найдена ни одна последовательность, возвращаем None





def process_1(sentences_1, sentences_2): # Первый этап обработки - простой поиск идеальных
    n_processed = 0
    for i_1, element in enumerate(sentences_1):
        i_2 = element[5]
        if sentences_2[i_2][5] == i_1:
            element[6] = True
            element[7] = "ИДЕАЛЕН"
            sentences_2[i_2][6] = True
            sentences_2[i_2][7] = "ИДЕАЛЕН"
            n_processed += 1
    return n_processed

def check_increase(sentences): # Проверка что номера ссылок без исключений увеличиваются от начала до конца текста. Ошибка может свидетельствовать о некачественном тексте.
    max_number = 0
    for element in sentences: 
        if element[6]:
            if max_number > element[5]:
                print(f'\nКритическая ошибка. Требование увеличения номеров соответствия нарушено.', max_number, element[5])
                sys.exit()
            max_number = element[5]
    return None


def removing_intersections(sentences_1, sentences_2): # 
    # Исключение выбросов.
    outlier_processed = 0
    while True:
        perfect_couples = []
        for i, element in enumerate(sentences_1):       # Формирование перечня идеальных пар
            if element[6]:
                perfect_couples.append([i, element[5]-i, element[5]])
                
        filled_data = fill_intermediate_data(perfect_couples)   # Заполнение мест неидеальных пар интерполированными данными. Для исключения ложных выбросов при больших пропусках.
        smoothed_data = smooth_data(filled_data)                # Сглаживание данных
        diff_betw_filled_smoothed = [[d1[0], abs(d1[1] - d2[1])] for d1, d2 in zip(filled_data, smoothed_data)] # Разница между интерполированными и сглаживанными данными
        sorted_data = sorted(diff_betw_filled_smoothed, key=lambda x: x[1], reverse=True)       # Сортируем список по второму значению в каждой строке в убывающем порядке
        max_outlier = sorted_data[0]        # Получаем первое значение из строки с максимальным вторым значением

        #graph_lib.draw_graph(diff_betw_filled_smoothed)
        if  max_outlier[1] > limit_of_max_outlier:
            i_1 = max_outlier[0]
            i_2 = sentences_1[i_1][5]
            sentences_1[i_1][6] = False
            sentences_1[i_1][7] = "ОТСЕЕН1"
            sentences_2[i_2][6] = False
            sentences_2[i_2][7] = "ОТСЕЕН1"
            outlier_processed += 1
        else:
            break
    # Исключение выбросов.

    # Нахождение пересечений
    number_of_couples = len(perfect_couples)
    matrix = [[0] * number_of_couples for _ in range(number_of_couples)]
    
    for i in range(number_of_couples):
        for j in range(number_of_couples):
            a1 = perfect_couples[i][1]
            a2 = perfect_couples[j][1]
            b1 = perfect_couples[i][0]
            b2 = perfect_couples[j][0]
            if a1 != a2:            # Предотвращение деления на 0
                # y=ax+b; a =i2-i1; y=i2; b=i1; x=[0,1]; a1x+b1 = a2x+b2; x=(b2-b1)/(a1-a2)
                x = (b2-b1)/(a1-a2)
                if 1 > x > 0:
                    matrix[i][j] = 1
    sums = [sum(row) for row in matrix]
    # Нахождение пересечений

    # Исключение элементов с пересечениями
    n_processed = 0
    for i, element in enumerate(sums):
        if element != 0:
            i_1 = perfect_couples[i][0]
            i_2 = perfect_couples[i][2]
            sentences_1[i_1][6] = False
            sentences_1[i_1][7] = "ОТСЕЕН2"
            sentences_2[i_2][6] = False
            sentences_2[i_2][7] = "ОТСЕЕН2"
            n_processed += 1
            
    # Проверка что пересечений не осталось, на всякий случай
    check_increase(sentences_1)
    check_increase(sentences_2)
    return outlier_processed, n_processed


def fill_intermediate_data(data): 
    data = [x[:2] for x in data]    # Отбрасываем лишние значения из списка.
    filled_data = []                # Создаем пустой список, в который будут добавлены строки с заполненными промежуточными значениями.
    for i in range(len(data) - 1):  # Начинаем цикл для обработки элементов исходной списка, за исключением последнего элемента.
        x1, y1 = data[i]            # Получаем значения первого элемента в паре (x, y)
        x2, y2 = data[i + 1]        # Получаем значения второго элемента в паре (x, y)
        x_range = range(x1, x2)     # Создаем диапазон значений между x1 и x2
        y_range = np.interp(x_range, [x1, x2], [y1, y2])                # Линейная интерполяция для y в диапазоне x_range
        filled_data.extend([[x, y] for x, y in zip(x_range, y_range)])  # Добавляем новые строки в filled_data
    filled_data.append(data[-1])    # Добавляем последнюю строку из исходного списка в filled_data, чтобы сохранить последнее значение без изменений.
    return filled_data

def smooth_data(data): 
    window_size = 2  # Размер окна скользящего среднего
    smoothed_data = []
    for i in range(len(data)):
        if i < window_size or i >= len(data) - window_size:
            smoothed_data.append(data[i])       # Сохраняем значения на границах без изменений
        else:
            values = [x[1] for x in data[i - window_size:i + window_size + 1]]
            smoothed_value = sum(values) / len(values)  # Вычисляем среднее значение в окне
            smoothed_data.append([data[i][0], smoothed_value])
    return smoothed_data


def process_3(sentences_1, sentences_2): # Третий этап обработки, поиск элементов с одной неверной ссылкой
    n_processed = 0
    for i, element in enumerate(sentences_1):
        if not element[6]:
            conditions_are_met = False
            i_2 = element[5]
            if i == 0:
                if sentences_1[i+1][6]:                         # Окружающие сопоставлены
                    if 0 == i_2 == sentences_1[i+1][5] - 1:
                        conditions_are_met = True
            elif i == len(sentences_1) - 1:
                if sentences_1[i-1][6]:                         # Окружающие сопоставлены
                    if sentences_1[i-1][5] + 1 == i_2 == len(sentences_2) - 1:
                        conditions_are_met = True
            else:
                if sentences_1[i-1][6] and sentences_1[i+1][6]:                         # Окружающие сопоставлены
                    if sentences_1[i-1][5] + 1 == i_2 == sentences_1[i+1][5] - 1:
                        conditions_are_met = True
            if conditions_are_met:
                distance = 1 - cosine(element[1], sentences_2[i_2][1])
                if distance > 0.35:
                    element[6] = True
                    element[7] = "ТРЕТИЙ "
                    sentences_2[i_2][5] = i
                    sentences_2[i_2][6] = True
                    sentences_2[i_2][7] = "ТРЕТИЙ "
                    n_processed += 1
    return n_processed





# ----------------- Блок создания объединенного списка со ссылками -------------------
# final_list[i]     i - номер объединненного абазаца
# final_list[i][t]  t - номер текста
# final_list[i][t][0] - начало строки (ссылка на sentences)
# final_list[i][t][1] - конец строки (ссылка на sentences)
# final_list[i][t][2] - количество токенов
# final_list[i][t][3] - количество абзацев
def creation_of_final_list(sentences): # 
    # Объединение строк с сомнительными, так чтобы вокруг сомнительных всегда были хорошо подобранные пары.
    # Обнаружение мест когда после хорошо подобранной пары без каких-либо разрывов идет другая хорошая пара.
    # Считаем такоме место концом одной строки и началом другой
    final_list = []
    next_start = [0, 0]
    for i, element in enumerate(sentences[0][:-1]):     # Цикл для обработки элементов, за исключением последнего элемента.
        if element[6]:                                  # Если у текущего предложения есть хорошая пара
            if sentences[0][i+1][6]:                    # и у следующего предложения тоже
                i_2 = element[5]                        # 
                if  sentences[0][i+1][5] == i_2 + 1:    # и каждый элемент пар находятся вплотную
                    final_list.append( [ [next_start[0], i  , 0, 0], 
                                         [next_start[1], i_2, 0, 0] ] )  # добавляем в сводный список
                    next_start = [i+1, i_2+1]
    final_list.append( [ [next_start[0], len(sentences[0])-1, 0, 0], 
                         [next_start[1], len(sentences[1])-1, 0, 0] ] )  # Добавляем конец текста
    
    for line in final_list: 
        update_tokens_and_paragraphs(line, sentences)
        
    print("Начал. формирование.", find_max_tokens(final_list)) 
    
    # Объединение предложений в абзацы
    combining_into_paragraphs(final_list, sentences)
    #print("Параграфы объединены.  ", find_max_tokens(final_list))
    combining_if_completion_is_not_normal(final_list, sentences)
    #print("Аномальные параграфы обработаны", find_max_tokens(final_list))
    combining_if_next_paragraph_is_dialogue(final_list, sentences)
    print("Доп. обработка.     ", find_max_tokens(final_list))
    
    # График количества токенов.
    #graph_lib.draw_nubmer_list(get_tokens_list(final_list)[0], 0)
    
    return final_list

# ---Объединение
# Объединение предложений в абзацы
def combining_into_paragraphs(final_list, sentences):
    i = 0
    while i < len(final_list) - 1:
        end_index_0     = final_list[i][0][1]
        end_index_1     = final_list[i][1][1]

        is_in_limit = limit_check(final_list, i)
        # Если последние предложения в обоих строках имеют признак конца абзаца оставляем как есть
        # Иначе - объединяем со следующей строкой
        if (sentences[0][end_index_0][2] and sentences[1][end_index_1][2]) or not is_in_limit:
            i += 1
        else:
            merge_lines(final_list, i, sentences)

# Объединение если абзац закончился не на точку, восклицательный или вопросительный знак, а, например, на двоеточие.
def combining_if_completion_is_not_normal(final_list, sentences):
    i = 0
    while i < len(final_list) - 1:
        end_index_0     = final_list[i][0][1]
        end_index_1     = final_list[i][1][1]
        last0 = sentences[0][end_index_0][0] 
        last1 = sentences[1][end_index_1][0] 
        sentence_completion_is_normal = (last0.endswith('.') or last0.endswith('!') or last0.endswith('?') or last0.endswith('”')) and (last1.endswith('.') or last1.endswith('!') or last1.endswith('?') or last1.endswith('”'))
        is_in_limit = limit_check(final_list, i)

        if sentence_completion_is_normal or not is_in_limit:
            i += 1
        else:
            merge_lines(final_list, i, sentences)

# Объединение если следующий абзац - диалоговый.
def combining_if_next_paragraph_is_dialogue(final_list, sentences):
    i = 0
    while i < len(final_list) - 1:
        end_index_0     = final_list[i][0][1]
        end_index_1     = final_list[i][1][1]
        next0 = sentences[0][end_index_0 + 1][0] 
        next1 = sentences[1][end_index_1 + 1][0] 
        sentence_is_dialogue = (next0.startswith('-') or next0.startswith('—') or next0.startswith('“')) and (next1.startswith('-') or next1.startswith('—') or next1.startswith('“'))
        is_in_limit = limit_check(final_list, i)

        if not sentence_is_dialogue or not is_in_limit:
            i += 1
        else:
            merge_lines(final_list, i, sentences)

def limit_check(final_list, i):
    return ((final_list[i][0][2] + final_list[i+1][0][2] < goal_tokens_limit -2) and (final_list[i][1][2] + final_list[i+1][1][2] < goal_tokens_limit -2))

def merge_lines(final_list, i, sentences):
    final_list[i][0][1]  = final_list[i+1][0][1]
    final_list[i][1][1]  = final_list[i+1][1][1]
    final_list.pop(i + 1)
    update_tokens_and_paragraphs(final_list[i], sentences)
# ---Объединение

# ---Обновление количества токенов заданной строке списка final_list. Обновляются для обоих текстов.
def update_tokens_and_paragraphs(line, sentences):
    text = get_text(line, sentences)
    number_of_paragraphs = [0, 0]
    for t in range(2):
        tokens = len(tokenizer_goal.encode(text[t]))
        if tokens > goal_tokens_limit:
            print(f'\n\n\nКритическая ошибка. Переполнение выходных токенов:', tokens, ", строка:")
            print(text[t])
            caller = inspect.currentframe().f_back.f_code.co_name
            print(f"\nФункция, вызвавшая ошибку: {caller}")
            sys.exit()
        line[t][2] = tokens
        
        start   = line[t][0] #- начало строки
        end     = line[t][1] #- конец строки
        for i in range(start, end + 1):
            if sentences[t][i][2]:       # Если предложение было помечено как последнее в абзаце
                number_of_paragraphs[t] += 1
        line[t][3] = number_of_paragraphs[t]

# Полчение текста книги по строке из final_list
def get_text(line, sentences):
    text = ['', '']
    for t in range(2):
        start   = line[t][0] #- начало строки
        end     = line[t][1] #- конец строки
        text[t] = get_one_paragraph_of_text(sentences, start, end, t)
    return text

# Полчение текста книги по интервалу номеров предложений
def get_one_paragraph_of_text(sentences, start, end, book):
    text = ''
    for i in range(start, end + 1):
        text += sentences[book][i][0]   # Последовательно добавляем предложения
        if sentences[book][i][2]:       # Если предложение было помечено как последнее в абзаце
            text += "\n"                # добавляем перенос строки
        else:
            text += " "                 # иначе добавляем пробел
    text = text.rstrip()  # удаляем лишние символы
    return text
# ---Обновление количества токенов заданной строке списка final_list. Обновляются для обоих текстов.
# ----------------- Блок создания объединенного списка со ссылками -------------------




# Объединение абзацев где мало токенов
def combining_short_paragraphs(final_list, sentences):
    iteration = 0
    while True:
        iteration += 1
        tokens_list, t = get_tokens_list(final_list)
        
        
        
        
        #print(tokens_list, t)
        
        
        
        
        start_index, end_index = find_optimal_limited_sequence(tokens_list)
        
        
        #print(tokens_list, t)
        #print(start_index, end_index)
        
        #input("enter...")
        
        
        

        if start_index < 0:
            print(f'Объединение коротких завершено. Итераций {iteration}')
            print("Абзацы, где мало токенов объединены.", find_max_tokens(final_list)) 
            break
        
        # Объединение абзацев с малым количеством токенов
        # Добавление к первому абзацу остальных
        final_list[start_index][0][1] = final_list[end_index][0][1]
        final_list[start_index][1][1] = final_list[end_index][1][1]
        # Пересчет токенов по объединяемым абзацам
        update_tokens_and_paragraphs(final_list[start_index], sentences)
        # Удаление добавленных абзацев
        del final_list[start_index+1:end_index+1]
        # Объединение абзацев с малым количеством токенов
        
        text = get_text(final_list[start_index], sentences)
        
        #print("==================")
        #print(text[t])
        #print("==================")
        #print("Всего токенов:", final_list[start_index][t][2])
        #print(find_max_tokens(final_list))
        
        # График количества токенов.
        #graph_lib.draw_nubmer_list(get_tokens_list(final_list)[0], iteration)

# Поиск абзацев с максимальным количеством токенов и получение средней величены токенов по всем абзацам 
def find_max_tokens(lst):
    max_index       = [-1, -1]
    max_value       = [-1, -1]
    column_sum      = [0, 0]
    column_count    = [0, 0]
    average         = [0, 0]
    for t in range(2):
        for i, row in enumerate(lst):
            column_sum[t] += row[t][2]
            column_count[t] += 1
            if row[t][2] > max_value[t]:
                max_value[t] = row[t][2]
                max_index[t] = i
        average[t] = column_sum[t] / column_count[t]
    return ['Максимальные количества токенов:', max_value, 'средние значения:', average]


# Нахождение самой длинной последовательности чисел в списке, с заданными ограничениями:
# 1. Среднее значение минимально;
# 2. Сумма последовательности не более:
#     - трети от максимального числа токенов целевой модели без дополнительных требований;
#     - двух третей от максимального числа токенов целевой модели если среднее значение меньше 50 или при увеличении лимита на один шаг последовательность увеличилась на 2 или более;
#     - максимального числа токенов целевой модели в любых случаях если среднее значение меньше 50 или при увеличении лимита на один шаг последовательность увеличилась на 4 или более;
def find_optimal_limited_sequence(tokens_list):
    one_third_of_limit = goal_tokens_limit/3
    two_third_of_limit = 2*goal_tokens_limit/3

    prev_start_index, prev_end_index, prev_length = -1, -1, 0
    best_start_index, best_end_index, best_length = -1, -1, 0
    total_tokens = 0
    current_limit_avg = 0
    if sum(tokens_list) < goal_tokens_limit:
        return best_start_index, best_end_index
    while total_tokens < goal_tokens_limit:
        new_start_index, new_end_index, new_length, total_tokens = find_longest_avg_limited_sequence(tokens_list, current_limit_avg)
        
        if (new_start_index != prev_start_index or new_end_index != prev_end_index) \
            and     (total_tokens < one_third_of_limit and  new_length > prev_length \
                or   total_tokens < two_third_of_limit and (new_length > prev_length + 1 or current_limit_avg <= 50) \
                or   total_tokens < goal_tokens_limit  and (new_length > prev_length + 2 or current_limit_avg <= 40) ):
            best_start_index = new_start_index
            best_end_index = new_end_index
            best_length = new_length 
                
        prev_start_index = new_start_index
        prev_end_index = new_end_index
        prev_length = new_length 
        current_limit_avg += 1
    return best_start_index, best_end_index



# Получение отдельного списков токенов
# Извлекается та часть где больше среднее
def get_tokens_list(final_list):
    tokens_list = [[], []]
    sum_all_tokens = [0, 0]
    for element in final_list:
        for n in range(2):
            tokens_list[n].append(element[n][2])
            sum_all_tokens[n] += element[n][2]
    average0 = sum_all_tokens[0] / len(final_list)
    average1 = sum_all_tokens[1] / len(final_list)
    #print(sum_all_tokens)
    #print(average0, average1)
    book = 0
    if average0 < average1:
        book = 1
    return tokens_list[book], book



# Функция находит самую длинную последовательность чисел в списке numbers, среднее значение которой меньше limit_avg. 
# Вариант 3, с пробитием барьера
# Функция возвращает кортеж, содержащий индексы начала и конца найденной последовательности.
def find_longest_avg_limited_sequence(numbers, limit_avg):
    # Инициализируем переменные для хранения наилучшей последовательности
    best_start_idx = 0      # Индекс начала наилучшей последовательности
    best_len = 0            # Длина наилучшей последовательности
    best_sum = 0            # Сумма элементов наилучшей последовательности
    best_avg = float('inf') # Среднее значение элементов наилучшей последовательности
    
    for start_idx  in range(len(numbers)):
        found_len  = 0
        found_sum  = 0
        found_avg  = float('inf')
        cur_len  = 0
        cur_sum  = 0
    
        for cur_idx   in range(start_idx , len(numbers)):
            cur_sum += numbers[cur_idx]     # Добавляем текущий элемент к сумме
            cur_len += 1                    # Увеличиваем длину подпоследовательности
            cur_avg  = cur_sum  / cur_len   # Вычисляем среднее значение элементов подпоследовательности
            # Обновляем найденные наилучшие значения, если средняя величина последовательности не превышает заданного лимита
            if cur_avg  <= limit_avg:       
                found_len = cur_len 
                found_sum = cur_sum
                found_avg = cur_avg 
        # Если текущая последовательность превосходит лучшую по всем параметрам, обновляем наилучшие индексы и значения
        if  (found_len   > best_len  ) or \
            (found_len  == best_len   and found_avg  < best_avg) :
            best_start_idx  = start_idx 
            best_len    = found_len 
            best_sum    = found_sum
            best_avg    = found_avg 
    if best_len > 1:
        best_end_idx = best_start_idx + best_len - 1
    else:
        best_start_idx  = -1
        best_end_idx    = -1
    #print(f'Лимит: {limit_avg}, начало: {best_start_idx}, длина: {best_len }, сумма {best_sum}, среднее: {best_avg:.2f}')
    return best_start_idx , best_end_idx, best_len, best_sum
    
    




# Объединение абзацев в блоки текста
def combining_paragraphs_into_blocks(final_list, sentences):

    tokens_list, book = get_tokens_list(final_list)
    
    #print(tokens_list)
    
    solution, result_list = solver(tokens_list, goal_tokens_limit)
    
    #print(solution, result_list)


    iteration = 0

    for element in solution:
        start_index = iteration
        end_index = start_index + element - 1
        # Объединение абзацев с малым количеством токенов
        # Добавление к первому абзацу остальных
        final_list[start_index][0][1] = final_list[end_index][0][1]
        final_list[start_index][1][1] = final_list[end_index][1][1]
        # Пересчет токенов по объединяемым абзацам
        update_tokens_and_paragraphs(final_list[start_index], sentences)
        # Удаление добавленных абзацев
        del final_list[start_index+1:end_index+1]
        # Объединение абзацев с малым количеством токенов
        
        
        iteration += 1
        
    tokens_list, book = get_tokens_list(final_list)
    
    
    #graph_lib.draw_nubmer_list(get_tokens_list(final_list)[0], 'end')
    
    #print(tokens_list)
    print(find_max_tokens(final_list)) 


