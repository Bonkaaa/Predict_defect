import numpy as np
import pandas as pd
import sklearn.metrics
from IPython.core.pylabtools import figsize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

df_train = pd.read_csv(r'D:\ML\Finding_defect\Files\Python\train.csv')
df_test = pd.read_csv(r'D:\ML\Finding_defect\Files\Python\test.csv')

df_test.drop(df_test.columns[0:3], axis =1, inplace= True)

df_train.drop(df_train.columns[0:3], axis =1, inplace= True)

for i in df_train.columns:
    percentile25 = df_train[i].quantile(0.25)
    percentile75 = df_train[i].quantile(0.75)
    iqr = percentile75 - percentile25
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr
    df_train = df_train.drop(df_train.loc[(df_train[i] > upper_limit) | (df_train[i] < lower_limit)].index)

y_train = df_train['bug']
y_test = df_test['bug']
df_train.drop(['bug'], axis=1, inplace= True)
df_test.drop(['bug'], axis=1, inplace= True)
X_train = df_train
X_test = df_test

pt = PowerTransformer()
pX_train = pt.fit_transform(X_train)
pX_test = pt.transform(X_test)

sc = StandardScaler()
zX_train = sc.fit_transform(pX_train)
zX_test = sc.transform(pX_test)

model = LogisticRegression(solver = 'saga', class_weight='balanced', random_state=42, max_iter=10000, penalty= 'l1', C = 0.01)
model.fit(zX_train, y_train)

y_predict = model.predict(zX_test)


print('AUC=', sklearn.metrics.roc_auc_score(y_test, y_predict))

print('Precision=', sklearn.metrics.precision_score(y_test, y_predict))

print('Recall=', sklearn.metrics.recall_score(y_test, y_predict))

print('F1=', sklearn.metrics.f1_score(y_test, y_predict))