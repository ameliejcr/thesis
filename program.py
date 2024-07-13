from datetime import datetime

from model_structure import Model
import excel_stuff

# model_test = Model(2, 2, 2)
# model_test.bottom.nodes[0].excitation = 0.23
# model_test.bottom.nodes[1].excitation = 0.26
# model_test.bottom.nodes[0].setWeight(model_test.middle.nodes[0], 0.45)
# model_test.bottom.nodes[0].setWeight(model_test.middle.nodes[1], 0.45)
# model_test.bottom.nodes[1].setWeight(model_test.middle.nodes[0], 0.49)
# model_test.bottom.nodes[1].setWeight(model_test.middle.nodes[1], 0.49)
#
# model_test.initialSettlingPhase()
# x = 1

print(datetime.now())
start = datetime.now()
model = Model(544, 500, 465)
# for i in range(608):
#     model.process(excel_stuff.phon_lib_train[i], excel_stuff.sem_lib_train[i])
for i in range(15):
    model.process(excel_stuff.phon_lib_test[i])
for i in range(15):
    model.process((excel_stuff.sem_lib_test[i]))

duration = datetime.now() - start
print(datetime.now())
print(duration)

