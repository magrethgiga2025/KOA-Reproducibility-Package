# Usage (from code/): python make_extra_figures.py [results_dir] [figure_dir]
"""Figures for Section 5 additions: convergence and parameter sensitivity.
No titles; panels labelled (a), (b), ... in the x-axis label; KOA in blue."""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
DATA=sys.argv[1] if len(sys.argv)>1 else "results"; OUT=sys.argv[2] if len(sys.argv)>2 else "figures"; os.makedirs(OUT,exist_ok=True)
INK="#0b0b0b"; INK2="#52514e"; GRID="#e4e3de"
plt.rcParams.update({"font.family":"serif","font.serif":["STIXGeneral","DejaVu Serif"],"mathtext.fontset":"stix",
 "font.size":8.5,"axes.labelsize":8.5,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,
 "axes.edgecolor":INK2,"axes.labelcolor":INK,"xtick.color":INK2,"ytick.color":INK2,"axes.linewidth":0.6,
 "axes.spines.top":False,"axes.spines.right":False,"savefig.bbox":"tight","savefig.pad_inches":0.02})
FULL_W=6.85
def save(fig,name):
    fig.savefig(os.path.join(OUT,name+".pdf")); fig.savefig(os.path.join(OUT,name+".png"),dpi=300); plt.close(fig); print("wrote",name)
ren=lambda s:s.replace("KOA-II","KOA").replace("AGE-MOEA2","AGE-MOEA-II")
# ---------------- convergence
STY={"KOA":("#2a78d6","-",2.0),"NSGA-II":("#eb6834","--",1.0),"NSGA-III":("#1baf7a","-.",1.0),
     "MOEA/D":("#eda100",":",1.2),"RVEA":("#e87ba4","--",1.0),"AGE-MOEA-II":("#008300","-.",1.0),
     "CMOPSO":("#6250d6",":",1.2),"SMS-EMOA":("#e34948","--",1.0),"GDE3":("#52514e","-.",1.0)}
c=pd.read_csv(os.path.join(DATA,"convergence_trajectories.csv")); c["algorithm"]=c.algorithm.map(ren)
g=c.groupby(["problem","algorithm","generation"]).IGD.mean()
probs=["DTLZ2","DTLZ7","WFG6","WFG9","MaF4","LSMOP9"]
fig,axes=plt.subplots(2,3,figsize=(FULL_W,4.4))
for k,(ax,p) in enumerate(zip(axes.flat,probs)):
    for a in list(STY)[1:]+["KOA"]:
        s=g.loc[(p,a)]; col,ls,lw=STY[a]
        ax.plot(s.index,s.values,color=col,ls=ls,lw=lw,zorder=3 if a=="KOA" else 2)
    ax.set_yscale("log"); ax.grid(True,which="major",color=GRID,lw=0.5); ax.set_xlim(0,300)
    ax.set_xlabel(f"Generation\n({'abcdef'[k]}) {p}",labelpad=2)
    if k%3==0: ax.set_ylabel("Mean IGD")
h=[Line2D([0],[0],color=STY[a][0],ls=STY[a][1],lw=STY[a][2],label=a) for a in STY]
fig.legend(handles=h,loc="upper center",ncol=9,frameon=False,bbox_to_anchor=(0.5,1.04),handlelength=2.2,columnspacing=1.0,fontsize=7.3)
fig.tight_layout(h_pad=1.2); save(fig,"fig_convergence")
# ---------------- sensitivity (median IGD change vs default, %)
s=pd.read_csv(os.path.join(DATA,"sensitivity_koa2.csv")); s["v"]=s.value.astype(str).str.replace(r"\.0$","",regex=True)
s=s.drop_duplicates(["problem","param","v","seed"])
PROB=["ZDT3","DTLZ2","DTLZ7","WFG4","LSMOP9","MaF4"]
PC={"ZDT3":("#eb6834","s"),"DTLZ2":("#1baf7a","^"),"DTLZ7":("#eda100","D"),"WFG4":("#e87ba4","v"),"LSMOP9":("#6250d6","P"),"MaF4":("#2a78d6","o")}
PARS=[("H",["3","5","10","20"],"5",r"$H$"),("archive_ratio",["0.5","1","1.5","2"],"1",r"$N_{\mathcal{A}}/N$"),
      ("beta",["0.05","0.1","0.2","0.4"],"0.1",r"$\beta$"),("q_lr",["0.1","0.3","0.5","0.7"],"0.3",r"$\lambda$")]
D=s[s.param=="default"].groupby("problem").IGD.median()
fig,axes=plt.subplots(1,4,figsize=(FULL_W,2.25),sharey=True)
for k,(ax,(par,vals,dv,lab)) in enumerate(zip(axes,PARS)):
    ax.axhline(0,color=INK2,lw=0.6); ax.axvline(vals.index(dv),color=GRID,lw=4,zorder=0)
    for p in PROB:
        y=[]
        for v in vals:
            if v==dv: y.append(0.0)
            else: y.append((s[(s.param==par)&(s.v==v)&(s.problem==p)].IGD.median()/D[p]-1)*100)
        col,mk=PC[p]; ax.plot(range(4),y,color=col,marker=mk,ms=4,lw=1.2,mec="white",mew=0.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(vals); ax.set_xlabel(f"({'abcd'[k]}) {lab}")
    ax.grid(True,axis="y",color=GRID,lw=0.5)
axes[0].set_ylabel("Change in median IGD (%)")
h=[Line2D([0],[0],color=PC[p][0],marker=PC[p][1],ms=4,lw=1.2,mec="white",label=p) for p in PROB]
fig.legend(handles=h,loc="upper center",ncol=6,frameon=False,bbox_to_anchor=(0.5,1.08))
fig.tight_layout(w_pad=0.6); save(fig,"fig_sensitivity")
