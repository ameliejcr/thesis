from datetime import datetime
from model_structure import Model
import excel_stuff

def compare(phon_list=None, sem_list=None, expected=None):
    if sem_list is not None:
        new_output = model.get_closest(sem_list)
        correct_sem = 0
        for i in range(len(new_output)):
            if new_output[i] == expected[i]:
                correct_sem += 1
        correct_per_sem = correct_sem / 144
        return correct_per_sem
    else:
        correct_phon = 0
        for i in range(len(phon_list)):
            diff = abs(phon_list[i] - expected[i])
            if diff < 0.0001:
                correct_phon += 1
        correct_per_phon = correct_phon / 400
        return correct_per_phon


print(datetime.now())
start = datetime.now()
model = Model(544, 500, 465, 144)
for n in range(1):
    for i in range(608):
        print(f"Learning word {n}.{i}")
        model.train(excel_stuff.phon_lib_train[i], excel_stuff.sem_lib_train[i])
model.remove_random_nodes(0.8)
for i in range(15):
    phon_result = model.predict(excel_stuff.phon_lib_test[i], True)
    result = compare(phon_result, None, excel_stuff.sem_lib_test[i])
    print(f"Phonetic Word {i}: {result}")
for i in range(15, 30):
    sem_result = model.predict(excel_stuff.sem_lib_test[i], False)
    result = compare(None, sem_result, excel_stuff.phon_lib_test[i])
    print(f"Semantic Word {i}: {result}")


# model_test = Model(20, 10, 5, 10)
# for n in range(10000):
#     for i in range(10):
#         model_test.train(excel_stuff.input_list_phon_train[i], excel_stuff.input_list_sem_train[i])
# for i in range(5):
#     phon_result = model_test.predict(excel_stuff.input_list_phon_test[i], True)
#     result = compare(phon_result, None, excel_stuff.input_list_sem_test[i])
#     print(f"Phonetic Word {i}: {result}")
# for i in range(5, 10):
#     sem_result = model_test.predict(excel_stuff.input_list_sem_test[i], False)
#     result = compare(None, sem_result, excel_stuff.input_list_phon_test[i])
    print(f"Semantic Word {i}: {result}")

duration = datetime.now() - start
print(datetime.now())
print(duration)
