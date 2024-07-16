from datetime import datetime
from model_structure import Model
import excel_stuff

print(datetime.now())
start = datetime.now()
model = Model(544, 500, 465, 144)
for n in range(1):
    for i in range(608):
        print(f"Learning word {n}.{i}")
        model.train(excel_stuff.phon_lib_train[i], excel_stuff.sem_lib_train[i])
model.remove_random_nodes(0.6)
for i in range(15):
    result = model.predict(excel_stuff.phon_lib_test[i], True, excel_stuff.sem_lib_test[i])
    print(f"Phonetic Word {i}: {result}")
for i in range(15, 30):
    result = model.predict(excel_stuff.sem_lib_test[i], False, excel_stuff.phon_lib_test[i])
    print(f"Semantic Word {i}: {result}")


# model_test = Model(20, 10, 5, 10)
# for n in range(1050):
#     for i in range(10):
#         model_test.train(excel_stuff.input_list_phon_train[i], excel_stuff.input_list_sem_train[i])
# for i in range(5):
#     result = model_test.predict(excel_stuff.input_list_phon_test[i], True, excel_stuff.input_list_sem_test[i])
#     print(f"Phonetic Word {i}: {result}")
# for i in range(5, 10):
#     result = model_test.predict(excel_stuff.input_list_sem_test[i], False, excel_stuff.input_list_phon_test[i])
#     print(f"Semantic Word {i}: {result}")

duration = datetime.now() - start
print(datetime.now())
print(duration)
