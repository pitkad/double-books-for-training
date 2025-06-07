
"""
Разбиение списка с целыми числами на группы со следующими ограничениями:
- сумма чисел в группе не может превосходить заданное ограничение;
- минимальное количество групп;
- сумма чисел в разных руппах должны быть как можно более близки к среднему.

Альтернативное описание:
Решение варианта задачи о рюкзаке, где есть много рюкзаков отдинакового
размера и их нужно заполнить поровну. Предметы подаются последовательно,
рюкзаки тоже берутся последовательно, к предыдущему рюкзаку вернуться нельзя.
"""

import time
import sys

from constants_lib import backtrack_solver_time_limit

# ---------- основная функция --------------
def solver(numbers, limit):
    #start_time = time.time()
    solution = backtrack_knapsack_solver(numbers, limit)
    #print(solution)
    #print("Время выполнения: ", time.time() - start_time, "секунд")


    if not solution:

        solution = split_list_by_sum(numbers, limit)
        #print(solution)

    result_list = get_result_list(solution, numbers)
        
    return solution, result_list
# ---------- основная функция --------------

# ---------- backtrack_knapsack_solver --------------
def backtrack_knapsack_solver(numbers, limit):
    number_of_numbers = len(numbers)
    number_of_groups = 1

    while True:
        start_time = time.time()
        limit_list = process_list(numbers, limit)
        solutions = generate_all_solutions(number_of_groups, number_of_numbers, limit_list)
        
        right_solution = choosing_solution(numbers, solutions)
        
        if right_solution >= 0:
            return solutions[right_solution]
        else:
            number_of_groups += 1
            
        if time.time() - start_time > backtrack_solver_time_limit:
            print(f"Поиск оптимального решения остановлен по превышению лимита времени. \nПоследний этап занял: \n{time.time() - start_time} секунд")
            print(f"Будет реализовано приблизительное решение")
            return False

# Создание списка с ограничениями для генератора решений.
# Каждый элемент списка представляет собой количество элементов в списке numbers,
# начиная с элемента с темже индексом, сумма которых наиболее близко подходит к ограничению.
def process_list(numbers, limit):
    output_list = []
    for i in range(len(numbers)):
        count = 0
        total_sum = 0
        for num in numbers[i:]:
            if total_sum + num <= limit:
                count += 1
                total_sum += num
            else:
                break
        output_list.append(count)
    return output_list

# Генератор всех возможных решений
def generate_all_solutions(num_columns, target_sum, limit_list):
    solutions = []
    current_solution = [0] * num_columns

    def backtrack(column_index, remaining_sum):
        if column_index == num_columns:
            if remaining_sum == 0:
                solutions.append(current_solution.copy())
            return

        # Вычисляем сумму значений всех элементов слева
        left_sum = sum(current_solution[:column_index])
        max_element = limit_list[left_sum]

        for i in range(1, min(remaining_sum + 1, max_element + 1)):
            current_solution[column_index] = i
            backtrack(column_index + 1, remaining_sum - i)
            current_solution[column_index] = 0

    backtrack(0, target_sum)
    return solutions

# Выбор лучшего решения
def choosing_solution(numbers, solutions):
    maximum = -1
    max_i   = -1
    for i, el in enumerate(solutions):
        product_of_sums = sums_product(numbers, el)
        if product_of_sums > maximum:
            maximum = product_of_sums
            max_i = i
    return max_i

def sums_product(input_list, sublist_counts):
    product_of_sums = 1         # Начальное значение для перемножения сумм
    for count in sublist_counts:
        sublist = input_list[:count]    # Получаем срез исходного списка
        sublist_sum = sum(sublist)      # Считаем сумму элементов среза
        product_of_sums *= sublist_sum  # Перемножаем сумму среза на текущее значение произведения сумм
        input_list = input_list[count:] # Удаляем из исходного списка использованный срез
    return product_of_sums
# ---------- backtrack_knapsack_solver --------------


# ------------ приблизительное решение --------------
# Подсчет теоретического количества групп в списке, сумма которых не превышает заданного лимита.
def count_groups_ddd(numbers, limit):
    total = 0
    number_of_groups = 0
    for number in numbers:
        if total + number > limit:
            number_of_groups += 1
            total = number
        else:
            total += number
    if total > 0:
        number_of_groups += 1
    return number_of_groups


def count_groups(numbers, limit):
    groups = []
    current_group = []
    total = 0
    for number in numbers:
        if total + number <= limit:
            current_group.append(number)
            total += number
        else:
            groups.append(current_group)
            current_group = [number]
            total = number
    if current_group:
        groups.append(current_group)
    number_of_groups = len(groups)
    return number_of_groups, groups



def split_list_by_sum(lst, limit):
    number_of_groups, start_groups = count_groups(lst, limit)
    сумма_всех_значений = sum(lst)
    
    avg = сумма_всех_значений / number_of_groups

    # Жалкая попытка найти решение лучше
    # Я устал возиться с этой библиотекой. Оставляю в сыром виде 24.01.2024
    groups = []
    current_group = []
    total = 0
    group_n = 1
    all_2 = 0
    for number in lst:
        #if (avg*group_n - all_2) > 0: # > number/2 and total + number <= limit:   # : #
        if (avg*group_n - all_2) > 0 and total + number <= limit:   # : #
        #if (avg*group_n - all_2) > number/2: # and total + number <= limit:   # : #
            current_group.append(number)
            total += number
        else:
            group_n += 1
            groups.append(current_group)
            current_group = [number]
            total = number
        all_2 += number
    if current_group:
        groups.append(current_group)
    # Жалкая попытка найти решение лучше
    
    optim_groups(groups)
    list_of_sum = get_list_of_sum(groups)

    
    for element in list_of_sum:
        if element > limit:
            print(f'\n\n\nКритическая ошибка. Переполнение выходных токенов:', element, ", строка:")
            print(list_of_sum)
            sys.exit()
    
    # Если попытка взять лучшее решение не удалась, возвращаемся к начальному
    if number_of_groups != len(list_of_sum):
        optim_groups(start_groups)
        groups = start_groups
    
    
    solution = []
    for el in groups:
        solution.append(len(el))

    return solution


def get_list_of_sum(groups):
    list_of_sum = []
    for element in groups:
        list_of_sum.append(sum(element))
    return list_of_sum


def optim_groups(groups):
    is_move = True
    while is_move:
        list_of_sum = get_list_of_sum(groups)
        pairs = get_pairs_with_max_difference(list_of_sum)
        for pair in pairs:
            is_move = compare_and_move_list(groups, pair[0])
            if is_move:
                break


# Принимает список чисел и возвращает список подсписков. Каждый подсписок состоит из двух элементов: 
# индекса первого элемента и разницы между этим элементом и следующим. Подсписки сортируются по разнице в порядке убывания.
def get_pairs_with_max_difference(numbers):
    sublists = []
    for i in range(len(numbers) - 1):
        sublist = [i, abs(numbers[i] - numbers[i + 1])]
        sublists.append(sublist)
    return sorted(sublists, key=lambda x: x[1], reverse=True)


# Переписать!!!!
def compare_and_move_list(matrix, row_index):
    current_row = matrix[row_index]
    next_row = matrix[row_index + 1]

    # Считаем суммы элементов текущей и следующей строки
    current_sum = sum(current_row)
    next_sum = sum(next_row)

    first_is_bigger = True if current_sum > next_sum else False

    # Определяем индекс элемента для перемещения:
    #  - если сумма текущей строки больше, перемещаем последний элемент в следующую строку
    #  - если сумма следующей строки больше, перемещаем первый элемент в текущую строку
    if first_is_bigger:
        element_to_move = current_row.pop()
        next_row.insert(0, element_to_move)
    else:
        element_to_move = next_row.pop(0)
        current_row.append(element_to_move)

    # Пересчитываем суммы элементов после перемещения
    new_current_sum = sum(current_row)
    new_next_sum = sum(next_row)

    #print(new_current_sum * new_next_sum, '>', current_sum * next_sum)
    # Проверяем, что произведение сумм стало больше
    if new_current_sum * new_next_sum > current_sum * next_sum:
        return True
    else:
        # Если произведение сумм не увеличилось, возвращаем элемент на прежнее место
        if first_is_bigger:
            current_row.append(element_to_move)
            next_row.pop(0)
        else:
            current_row.pop()
            next_row.insert(0, element_to_move)

        return False
# ------------ приблизительное решение --------------


#---------------------------ОБЩИЕ--------------------
# Подсчет результата
def get_result_list(solution, numbers):
    result_list = []
    i = 0
    for el in solution:
        current_sum = 0
        for n in range(el):
            current_sum += numbers[i]
            i +=1
        result_list.append(current_sum)
    return result_list
#---------------------------ОБЩИЕ---------------------



