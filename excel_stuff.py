import openpyxl

wb_phon = openpyxl.load_workbook('eng_phonetic_frame.xlsx', True)
wb_sem = openpyxl.load_workbook('semantic_frame.xlsx', True)

ws_phon_train = wb_phon['Sheet1']
ws_sem_train = wb_sem['Sheet1']
ws_phon_test = wb_phon['Sheet2']
ws_sem_test = wb_sem['Sheet2']

phon_lib_train = []
sem_lib_train = []
phon_lib_test = []
sem_lib_test = []


for row in ws_phon_train.iter_rows(min_row=5,min_col=2, max_col=145, max_row=612, values_only=True):
    phon_lib_train.append(row)

for row in ws_sem_train.iter_rows(min_row=2, min_col=2, max_col=401, max_row=609, values_only=True):
    sem_lib_train.append(row)

for row in ws_phon_test.iter_rows(min_row=1, min_col=2, max_col=145, max_row=30, values_only=True):
    phon_lib_test.append(row)

for row in ws_sem_test.iter_rows(min_row=1, min_col=2, max_col=401, max_row=30, values_only=True):
    sem_lib_test.append(row)

