import numpy as np
import pandas as pd
from sklearn.svm import SVC

from definitions import ROOT_DIR
from src.SVMs import X, Y
data = pd.read_csv(ROOT_DIR + '/data/data_banknote_authentication.txt', header=None)

csv_x = data.iloc[:, :-1]
csv_y = data.iloc[:, -1:]

print('Pandas')
print(csv_x.to_numpy())
print(csv_y.to_numpy())
print('Original')


model = SVC(random_state=42)
model.fit(csv_x, csv_y.values.ravel())


print(model.predict([[-2.7908,-5.7133,5.953,0.45946]]))