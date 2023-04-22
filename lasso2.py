import numpy as np
import pandas as pd
from numpy import loadtxt
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score, precision_recall_curve
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn import svm
from tabulate import tabulate


df1 = pd.read_csv("MD_dispatch_2022_08_18_pivot1.csv")
df2 = pd.read_csv("Non_MD_dispacth_2022_08_19_pivot1.csv")

frames = [df1, df2]

df = pd.concat(frames,ignore_index=True)

print(df.shape)

df = df.fillna(0)

X = df.iloc[:,0:60]
y = df.iloc[:,60]



# Spilt into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

logreg = SelectFromModel(LogisticRegression(penalty='l1', C=1, solver='liblinear'))
logreg.fit(X_train, y_train)

selected_feat = X_train.columns[(logreg.get_support())]
print("total features: {}".format((X_train.shape[1])))
print("selected features: {}".format(len(selected_feat)))
print("features with coefficients shrank to zero: {}".format(np.sum(logreg.estimator_.coef_ == 0)))

X_train_selected = logreg.transform(X_train)
X_test_selected = logreg.transform(X_test)

LG = LogisticRegression(random_state=0)
LG.fit(X_train_selected,y_train)
y_pred=LG.predict(X_test_selected)

#create a classifier
#cls = svm.SVC(kernel="linear")
#train the model
#cls.fit(X_train_selected,y_train)
#predict the response
#y_pred = cls.predict(X_test_selected)

print("F1 Score (average=macro)",f1_score(y_test, y_pred, pos_label=0))
print("Precision Score (average=macro)",precision_score(y_test, y_pred, pos_label=0))
print("Recall Score (average=macro)",recall_score(y_test, y_pred, pos_label=0))


precision, recall, thresholds = precision_recall_curve(y_test, y_pred, pos_label=0)

plt.figure(figsize = (10,8))
plt.plot([0, 1], [0.5, 0.5],'k--')
plt.plot(recall, precision, label = 'lasso')
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('LASSO MB Replacement Not Required: PRC curve')
plt.savefig("lasso2.png")
plt.show()