import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams["font.family"]="DejaVu Serif"
W,H=4.8,4.3
fig=plt.figure(figsize=(W,H)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off")
steps=[("1. Corpus collection",["26 Vietnamese banks","Annual reports, 2015–2025","PDFs named by bank, year"]),
("2. OCR extraction",["Tesseract (Vietnamese)","300 dpi rendering","Grayscale, Gaussian","denoising, Otsu threshold"]),
("3. Pre-processing",["Unicode NFC, lowercase","Remove non-alphanumerics","Keep Vietnamese diacritics","Consolidate whitespace"]),
("4. Dictionary matching",["117 keywords/phrases","6 dimensions (UX, ST, PL,","OP, CU, DA)","Exact phrase matching","with word boundaries"]),
("5. Frequency count",["Match counts c(k, r)","Dimension frequency F","Document frequency df","TF and IDF weights"]),
("6. Index construction",["S_DTI: standardized","frequency","LnDTI: log frequency","W_DTI: TF-IDF weighted"])]
bw,bh,gx=1.46,1.55,0.14
x0=(W-3*bw-2*gx)/2
rows=[2.4,0.5]
pos=[]
for i,(t,b) in enumerate(steps):
    r,c=divmod(i,3); x=x0+c*(bw+gx); y=rows[r]
    ax.add_patch(FancyBboxPatch((x,y),bw,bh,boxstyle="square,pad=0",fc="white",ec="#333",lw=0.9))
    ax.add_patch(FancyBboxPatch((x,y+bh-0.36),bw,0.36,boxstyle="square,pad=0",fc="#e6e6e6",ec="#333",lw=0.9))
    ax.text(x+bw/2,y+bh-0.18,t,ha="center",va="center",fontsize=7.2,fontweight="bold")
    for j,l in enumerate(b):
        bullet = l[0].isupper() or l[0].isdigit() or l[0] in "SLW1"
        ax.text(x+0.07,y+bh-0.55-j*0.215,l,ha="left",va="center",fontsize=7)
    pos.append((x,y))
def arr(a,b): ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=9,lw=1,color="#333",shrinkA=0,shrinkB=0))
for i in (0,1): arr((pos[i][0]+bw,pos[i][1]+bh/2),(pos[i+1][0],pos[i+1][1]+bh/2))
for i in (3,4): arr((pos[i][0]+bw,pos[i][1]+bh/2),(pos[i+1][0],pos[i+1][1]+bh/2))
# wrap: from box 3 bottom to box 4 top
arr((pos[2][0]+bw/2,pos[2][1]),(pos[3][0]+bw/2,pos[3][1]+bh)) if False else None
ax.plot([pos[2][0]+bw/2,pos[2][0]+bw/2],[pos[2][1],2.28],color="#333",lw=1)
ax.plot([pos[2][0]+bw/2,pos[0][0]+bw/2],[2.28,2.28],color="#333",lw=1)
arr((pos[0][0]+bw/2,2.28),(pos[0][0]+bw/2,pos[3][1]+bh))
ax.text(W/2,H-0.12,"Text-mining pipeline for the disclosure-intensity index (DTI)",ha="center",va="center",fontsize=8,fontweight="bold")
ax.text(W/2,0.2,"Output: bank-year panel of DTI measures, merged with ROA, ROE, total assets and ownership",ha="center",va="center",fontsize=6.6,style="italic")
fig.savefig("figures/fig1_pipeline.png",dpi=600); fig.savefig("figures/fig1_pipeline.pdf"); fig.savefig("figures/fig1_pipeline.svg")
