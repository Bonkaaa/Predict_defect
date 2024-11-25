import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PowerTransformer

df_val = pd.read_csv(r'D:\ML\Finding_defect\Files\Cpp\val.csv')
df_val.describe()

df_val.drop(df_val.columns[0:3], axis =1, inplace= True)


for i in df_val.columns:
    percentile25 = df_val[i].quantile(0.25)
    percentile75 = df_val[i].quantile(0.75)
    iqr = percentile75 - percentile25
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr
    df_val = df_val.drop(df_val.loc[(df_val[i] > upper_limit) | (df_val[i] < lower_limit)].index)

y_val = df_val['bug']
df_val.drop(['bug'], axis=1, inplace= True)
X_val = df_val

pt = PowerTransformer()
pX_val = pt.fit_transform(X_val)

sc = StandardScaler()
zX_val = sc.fit_transform(pX_val)

model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=10000, solver='saga')

param_grid = {
    'C' : [0.01, 0.1, 1],
    'penalty' : ['l1', 'l2','elasticnet'],
    'l1_ratio': [0.2, 0.5, 0.8]
}

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
}

grid_search = GridSearchCV(estimator = model, param_grid = param_grid, scoring = scoring, cv = 5, refit = 'roc_auc')
grid_search.fit(zX_val, y_val)

best_params = grid_search.best_params_
print(best_params)

