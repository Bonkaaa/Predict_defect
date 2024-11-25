import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

df_train = pd.read_csv(r'D:\ML\Finding_defect\Files\Python\train.csv')
df_test = pd.read_csv(r'D:\ML\Finding_defect\Files\Python\test.csv')

df_test.drop(df_test.columns[0:3], axis=1, inplace=True)
df_train.drop(df_train.columns[0:3], axis=1, inplace=True)

y_train = df_train['bug']
y_test = df_test['bug']
df_train.drop(['bug'], axis=1, inplace=True)
df_test.drop(['bug'], axis=1, inplace=True)
X_train = df_train
X_test = df_test

model = LogisticRegression(solver = 'saga', class_weight='balanced', random_state=42, max_iter=10000, penalty= 'elasticnet', C = 0.01, l1_ratio=0.2)
model.fit(X_train, y_train)

y_predict = model.predict(X_test)



print('AUC=', sklearn.metrics.roc_auc_score(y_test, y_predict))

print('Precision=', sklearn.metrics.precision_score(y_test, y_predict))

print('Recall=', sklearn.metrics.recall_score(y_test, y_predict))

print('F1=', sklearn.metrics.f1_score(y_test, y_predict))