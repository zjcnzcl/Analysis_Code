from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from response_scoring import normalize_accuracy
p=argparse.ArgumentParser()
p.add_argument('--input',required=True)
p.add_argument('--output',default='figures')
args=p.parse_args()
FIG=PRIVATE=Path(args.output)
FIG.mkdir(parents=True,exist_ok=True)
data,_=normalize_accuracy(pd.read_excel(args.input,sheet_name='05_分析随机顺序'))

categories=['标准色卡','肤色','发绀','血液颜色','组织颜色']
category_labels=['Overall','Color card','Skin tone','Cyanosis','Blood color','Tissue color']
colors=['#3478F6','#F39800','#10B893','#9055EF']
markers=['o','s','^','D']
labels=['L1 Conventional LED','L2 LED (Ra 83; low R9)','L3 High-R9 LED','L4 Halogen reference']
plt.rcParams.update({'font.family':'Arial','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(11,6.2))
figure_stats=[]
for i,light in enumerate(['L1','L2','L3','L4']):
    means=[];ci=[]
    for task in [None]+categories:
        sel=data[data['光源编号'].eq(light)]
        if task is not None:sel=sel[sel['任务类型'].eq(task)]
        values=sel.groupby('受试者编号')['是否正确1_0'].mean()
        means.append(values.mean()*100)
        ci.append(student_t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))*100)
        figure_stats.append({'light':light,'task':task or 'Overall','mean_percent':means[-1],'half_ci_percent':ci[-1]})
    ax.errorbar(np.arange(6)+(i-1.5)*.17,means,yerr=ci,fmt=markers[i],color=colors[i],markersize=7,capsize=4,elinewidth=1.6,label=labels[i])
ax.set_xticks(range(6),category_labels);ax.set_ylim(0,103)
ax.set_ylabel('Recognition accuracy (%)');ax.set_xlabel('Simulated clinical color-recognition task')
ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True)
ax.legend(ncol=2,loc='lower center',bbox_to_anchor=(.5,1.02),frameon=False,fontsize=11)
fig.tight_layout();fig.savefig(FIG/'Figure3.png',dpi=300,bbox_inches='tight');fig.savefig(FIG/'Figure3.pdf',bbox_inches='tight');plt.close(fig)
pd.DataFrame(figure_stats).to_csv(PRIVATE/'Figure3_source_values.csv',index=False)
