import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv("MD_dispatch_2022_08_18_pivot1.csv")
df2 = pd.read_csv("Non_MD_dispacth_2022_08_19_pivot1.csv")

frames = [df1, df2]

df = pd.concat(frames,ignore_index=True)

class_1 = df[df['Dispatch'] == 0]
class_2 = df[df['Dispatch'] == 1]


fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(10, 4))
axs[0].hist(class_1['PC SN730 NVMe WDC 256GB'], alpha=0.5, label='No Dispatch')
axs[0].hist(class_2['PC SN730 NVMe WDC 256GB'], alpha=0.5, label='Dispatch ')
axs[0].set_xlabel('PC SN730 NVMe WDC 256GB')
axs[1].hist(class_1['USB'], alpha=0.5, label='No Dispatch')
axs[1].hist(class_2['USB'], alpha=0.5, label='Dispatch')
axs[1].set_xlabel('USB')
axs[2].hist(class_1['PM9A1 NVMe Samsung 512GB'], alpha=0.5, label='No Dispatch')
axs[2].hist(class_2['PM9A1 NVMe Samsung 512GB'], alpha=0.5, label='Dispatch')
axs[2].set_xlabel('PM9A1 NVMe Samsung 512GB')


axs[0].set_ylabel('Frequency')

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

# Adjust the layout of the subplots and legend
plt.subplots_adjust(right=0.85)
plt.tight_layout()

plt.savefig("Hist2.png")
# Show the plot
plt.show()