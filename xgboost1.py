import numpy as np
import pandas as pd
from numpy import loadtxt
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,precision_score,recall_score,precision_recall_curve


df1 = pd.read_csv("MD_dispatch_2022_08_18_pivot1.csv")
df2 = pd.read_csv("Non_MD_dispacth_2022_08_19_pivot1.csv")

frames = [df1, df2]

df = pd.concat(frames,ignore_index=True)

#print(df.describe())

#print(df.info())

print(df.shape)

df = df.fillna(0)

X = df.iloc[:,0:60]
y = df.iloc[:,60]

# Spilt into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# XGB Classifier
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, seed=123)
eval_set = [(X_train, y_train), (X_test, y_test)]

# Fit the classifier to the training set
xg_cl.fit(X_train, y_train, eval_metric=["error"], eval_set=eval_set, verbose=True)
results = xg_cl.evals_result()

# Predict the labels of the test set: preds
y_pred = xg_cl.predict(X_test)


print("F1 Score (average=macro)",f1_score(y_test, y_pred, average="macro"))
print("Precision Score (average=macro)",precision_score(y_test, y_pred, average="macro"))
print("Recall Score (average=macro)",recall_score(y_test, y_pred, average="macro"))


precision, recall, thresholds = precision_recall_curve(y_test, y_pred, pos_label=1)

plt.figure(figsize = (10,8))
plt.plot([0, 1], [0.5, 0.5],'k--')
plt.plot(recall, precision, label = 'Lasso')
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('XGBOOST MB Replacement Required: PRC curve')
plt.savefig("xgboost.png")
plt.show()

