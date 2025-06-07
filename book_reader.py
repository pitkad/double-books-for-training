import sys
sys.path.append('lib') # Указывает путь к папке с библиотеками
import fb2_reader_lib
import text_analysis_lib

import pickle
import csv
import json

from constants_lib import instruction

#from graph_lib import clear_image_folder
#clear_image_folder()  #Очищаем папку с графиками

json_data = []

print("\nСреда инициализирована")




folder_path = 'fb2_transformation' # Путь к папке с файлами .fb2

books = fb2_reader_lib.read_and_preprocessing_files(folder_path)





n_sections = fb2_reader_lib.initial_structure_check(books)
sections = list(zip(books[0].find_all('section'), books[1].find_all('section')))
    

for i in range(n_sections):    #   range(n_sections):         range(1, 2): 
    sentences_list = []
    print("Начата обработка гавы", i + 1)

    sentences_1 = text_analysis_lib.create_sentences_list(sections[i][0])     # Получение списка предложений главы с векторами и параметрами
    sentences_2 = text_analysis_lib.create_sentences_list(sections[i][1])
    print("\nВекторизация глав", i + 1,"завершена. Предложений в  1 тексте:        ", len(sentences_1),",       во 2 тексте:        ", len(sentences_2))
    
    sentences = [sentences_1, sentences_2]
    
    #with open('список' + str(i) + '_1.pickle', 'wb') as файл:
    #    pickle.dump(sentences_1, файл)
    #with open('список' + str(i) + '_2.pickle', 'wb') as файл:
    #    pickle.dump(sentences_2, файл)
        
        

    sentences_1, sentences_2 = text_analysis_lib.первоначальный_поиск_соответствий(sentences_1, sentences_2)
    print("Первоначальный поиск соответствий выполнен. этап. Обнаружено", len(sentences_1), "и ", len(sentences_2), "предложений.")
    
    # Первый этап обработки - простой поиск идеальных
    n_processed = text_analysis_lib.process_1(sentences_1, sentences_2)
    print("Первый этап. Обработано", n_processed, "из", len(sentences_1))
    
    # Второй этап обработки - исключение пересечений
    outlier_processed, n_processed = text_analysis_lib.removing_intersections(sentences_1, sentences_2)
    print("Второй этап. Отсеено по выбросам:", outlier_processed, ". Отсеено по пересечению:", n_processed, "из", len(sentences_1))


    
    n_processed  = text_analysis_lib.process_3(sentences_1, sentences_2)
    n_processed += text_analysis_lib.process_3(sentences_2, sentences_1)
    print("Третий этап. Добавлено", n_processed, "из", len(sentences_1))

    
    
    # Создание объединенного списка со ссылками
    final_list = text_analysis_lib.creation_of_final_list(sentences)
    print(f"Финальный лист сформирован. Получено {len(final_list)} абзацев.")
    # Объединение абзацев, где мало токенов
    # Например диалоги
    text_analysis_lib.combining_short_paragraphs(final_list, sentences)
    print(f"Короткие абзацы объединены. Осталось {len(final_list)} абзацев.")

    # Объединение абзацев в блоки текста
    text_analysis_lib.combining_paragraphs_into_blocks(final_list, sentences)
    print(f"Абзацы объединены в блоки. Осталось {len(final_list)} блоков.")
    


    '''
    # Печать и экспорт
    csv_filename = 'example1.csv'
    with open(csv_filename, 'w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        
        
        for index, row_from_final in enumerate(final_list):
            tokens  = [[],[]]
            text = text_analysis_lib.get_text(row_from_final, sentences)
            for t in range(2):
                tokens[t] = row_from_final[t][2] #- количество токенов
                if t:
                    print("--------------")
                else:
                    print("==================")
                print(index, row_from_final[t])
                print("Предложений:", row_from_final[t][1] - row_from_final[t][0] + 1)
                print(text[t])
                
            writer.writerow([ text[0], tokens[0], text[1], tokens[1] ])

    '''
    
    
    for index, row_from_final in enumerate(final_list):
        text = text_analysis_lib.get_text(row_from_final, sentences)
        json_data.append({"instruction": instruction ,"input": text[0], "output": text[1]})
    



book_title = books[0].find('title').get_text() + " - " + books[1].find('title').get_text() 
output_file_name = fb2_reader_lib.process_filename(book_title) + ".json"
output_path = folder_path + '/' + output_file_name

json_string = json.dumps(json_data, indent=4, ensure_ascii=False)
with open(output_path, "w", encoding="utf8") as output_file:
    output_file.write(json_string)

print('\nРезультат сохранен в файл:')
print(output_path)
