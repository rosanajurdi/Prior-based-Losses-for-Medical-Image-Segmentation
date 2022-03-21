'''
Created on March 17, 2022
@author: Rosana El Jurdi

dataset indicators:

spleen -> spleen dataset
patient006 -H1 -> Acdc first tissue
patient006 -H2 -> Acdc first tissue
patient018_10_0_1.npy_H3 -> Acdc third tissue
hippocampus_004_0_0.npy_H1 -> Hippocampus first tissue,
hippocampus_ H2 -> Hippocampus second tissue
prostate_00_0_0.npy_H1  -> prostate h1
prostate_17_0_10.npy_H2 -> prostate h2
la_005_0_10.npy_  atrium
colon.npy_ -> colon
isles ->  case_
107_0_49.npy_ -> wmh 7659
'''

import pandas as pd
import matplotlib.pyplot as plt
print("hi")


df = pd.read_csv('/Users/rosana.eljurdi/PycharmProjects/Prior-based-Losses-for-Medical-Image-Segmentation/BenchmarkResults.csv')

df[7657:8766]
df = df.loc[df['sz1'] != 0]
sampled_mean = []

key_note = {'isles':['case_'],'hippoH1':['hippocampus', 'H1'],'hipooH2':['hippocampus', 'H2'],
            'prostateP1': ['prostate', 'H1'], 'prostateP2':['prostate', 'H2'],'colon':['colon'],
            'acdc-H1': ['patient','H1'], 'acdc-H2': ['patient','H2'],'acdc-H3': ['patient','H3'],
            'spleen': ['spleen'], 'atrium': ['la_'] , 'wmh': ['wmh']}

COLOR_note = {'isles':'blue','hippoH1':'yellow','hipooH2':'green',
            'prostateP1': 'red', 'prostateP2':'purple','colon':'black', 'acdc-H1': 'orange',
              'acdc-H2': 'brown', 'acdc-H3': 'cyan', 'spleen':'magenta', 'atrium': 'lightcoral', 'wmh': 'grey'}

wmh_dataset = df
fig, ax = plt.subplots()

for n, Label in enumerate(['prostateP1', 'isles','prostateP2',  'hippoH1','hipooH2','acdc-H1', 'acdc-H2', 'acdc-H3', 'spleen', 'atrium', 'colon']):
    print(Label)
    LABEL = list(key_note[Label])
    dataset = df[df['id'].str.contains(LABEL[0])]
    dataset = dataset[dataset['id'].str.contains(LABEL[-1])]
    wmh_dataset = wmh_dataset.drop(dataset.index)
    sampled_mean = []
    sampled_size = []
    for i in range(100):
        sample = dataset.sample(n=50)
        sampled_mean.append(sample.mean()['Dice'])
        sampled_size.append(sample.mean()['Organ-Size'])
        ax.scatter(sampled_size,sampled_mean, c=COLOR_note[Label])
    print(LABEL)

for i in range(100):
    sample = wmh_dataset.sample(n=50)
    sampled_mean.append(sample.mean()['Dice'])
    sampled_size.append(sample.mean()['Organ-Size'])
    ax.scatter(sampled_size, sampled_mean, c=COLOR_note['wmh'])
# ax.legend()
#ax.grid(True)
plt.show()

print(dataset)




print(df['id'])

