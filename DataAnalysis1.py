import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv("MD_dispatch_2022_08_18_pivot1.csv")
df2 = pd.read_csv("Non_MD_dispacth_2022_08_19_pivot1.csv")

frames = [df1, df2]

df = pd.concat(frames, ignore_index=True)

#cols = df.iloc[:, :20].columns
#cols = df.iloc[:, 20:40].columns
cols = df.iloc[:, 40:60].columns
corr_matrix = df[cols].corrwith(df['Dispatch'])

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(corr_matrix.to_frame(), annot=True, cmap='coolwarm', ax=ax)

plt.tight_layout()
plt.savefig("HeatMap3.png")
plt.show()

