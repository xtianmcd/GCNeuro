import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

A = np.load('Documents/att3ntions93.npy')

b=[]
for i in range(16):
    b.append([])
for r in A:
    s = np.argsort(r)[:16]
    for i in range(len(s)):
        b[i].append(s[i])

c={}
for i,k in enumerate(b):
    c[i]={}
    for idx in k:
        if idx not in c[i].keys(): c[i][idx]=0
        else: c[i][idx]+=1

d={}
for k,v in c.items():
    for kk,vv in c[k].items():
        if kk not in d.keys(): d[kk]=0
        d[kk]+= ((100-k)/100)*(vv+1)

regions = {76:'left frontal pole', 75:'left supremarginal gyrus', 74:'left superior temporal gyrus', 77:'left temporal pole', 72:'left superior frontal gyrus', 73:'left superior parietal lobule', 78:'left transverse temporal gyrus', 79:'left insula', 71:'left rostral middle frontal gyrus', 70:'left ronstral anterior cingulate cortex', 69:'left precuneus', 68:'left precentral gyrus', 81:'right banks of the superior temporal sulcus', 67:'left posterior cingulate cortex', 66:'left postcentral gyrus', 83:'right caudal middle frontal gyrus'}

e = dict([(regions[k],v/max(d.values())) for k,v in d.items() if k in regions.keys()])
sv = sorted(e.values(), reverse=True)

f={}
for s in sv:
    for k,v in e.items():
        if v==s:
            f[k]=v

sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(6, 15))
# sns.set_color_codes("pastel")
pal = sns.cubehelix_palette(len(f.keys()))
# cmap = sns.cubehelix_palette(as_cmap=True)
sns.barplot(x=list(f.values()), y=list(f.keys()),
            label="Region Weighting", palette=np.array(pal[::-1]))
plt.savefig('Documents/attentions.png',bbox_inches='tight')
