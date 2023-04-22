import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,precision_score,recall_score,precision_recall_curve

df1 = pd.read_csv("MD_dispatch_2022_08_18_pivot1.csv")
df2 = pd.read_csv("Non_MD_dispacth_2022_08_19_pivot1.csv")

frames = [df1, df2]

df = pd.concat(frames,ignore_index=True)

df = df.fillna(0)

X = df.iloc[:,0:60].values
y = df.iloc[:,60].values

#print(X)

X_train, X_test, y_train, y_test = train_test_split(
                                                    X, 
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0)

print("X_train", X_train.shape)
print("y_train", y_train.shape)

print("X_test", X_test.shape)
print("y_test", y_test.shape)

SS = StandardScaler()
X_train = SS.fit_transform(X_train)
X_test = SS.transform(X_test)

print(X_test)

lda = LDA()

X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

#explained_varience = lda.explained_variance_ratio_
#feature_names_in = lda.feature_names_in_

#print(explained_varience)

#print(feature_names_in)

LG = LogisticRegression(random_state=0)
LG.fit(X_train,y_train)
y_pred=LG.predict(X_test)

print("F1 Score (average=macro)",f1_score(y_test, y_pred, average="macro"))
print("Precision Score",precision_score(y_test, y_pred, average="macro"))
print("Recall Score",recall_score(y_test, y_pred, average="macro"))


precision, recall, thresholds = precision_recall_curve(y_test, y_pred, pos_label=1)

plt.figure(figsize = (10,8))
plt.plot([0, 1], [0.5, 0.5],'k--')
plt.plot(recall, precision, label = 'Lasso')
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('LDA MB Replacement Required: PRC curve')
plt.savefig("LDA1.png")
plt.show()

