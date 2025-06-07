

instruction = 'Translate into Russian:'
#instruction = 'Translate into English:'

# Константы для fb2_reader_lib.py
write_to_a_text_file    = 0
print_a_tag_map         = 0



# Константы для text_analysis_lib.py
show_progress_bar       = 1     # Отображать 

batch_of_vectors        = 15
seq_length              = 4
print_token_overflow    = 0     # Печать сообщений о переполнении токенов (когда предложение больше 128 токенов)

limit_of_max_outlier    = 2     # Ограничение максимального выброса относительно скользящего среднего

goal_tokens_limit       = 626  # Ограничение количества токенов для целевой модели


# Константы для graph_lib.py
image_folder            = 'pic'


# Константы для knapsack_solver_lib
backtrack_solver_time_limit = 5 # Лимит времени на поиск идеального решения в секундах (в текущей реализации может занять 10 раз больше времени)