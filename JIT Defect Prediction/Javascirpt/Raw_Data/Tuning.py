import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.model_selection import GridSearchCV

df_val = pd.read_csv(r'D:\ML\Finding_defect\Files\Javascript\val.csv')

df_val.drop(df_val.columns[0:3], axis =1, inplace= True)
y_val = df_val['bug']
df_val.drop(['bug'], axis=1, inplace= True)
X_val = df_val

model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=10000, solver='saga')

param_grid = {
    'C': [0.01, 0.1, 1],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'l1_ratio': [0.2, 0.5, 0.8]
}
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
}

grid_search = GridSearchCV(estimator = model, param_grid = param_grid, scoring = scoring, cv = 5, refit = 'roc_auc', error_score = 'raise')
grid_search.fit(X_val, y_val)

best_params = grid_search.best_params_
print(best_params)