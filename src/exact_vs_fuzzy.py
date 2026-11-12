import re, unicodedata, subprocess, glob, pandas as pd, numpy as np, openpyxl
import os
REPORTS=os.environ.get("REPORTS_DIR","reports")          # folder with NVB_15CN_BCTN.pdf ... (annual-report PDFs, not in this repo)
kw=pd.read_csv("data/dictionary_120_terms.csv"); terms=list(zip(kw.dimension,kw.keyword))
dd={t:dim for dim,t in terms}
print("unique strings",len(dd),"multi-word",sum(' ' in t for t in dd))
def norm(s):
    s=unicodedata.normalize("NFC",s).lower(); s=re.sub(r"[^0-9a-zà-ỹ]+"," ",s); return re.sub(r"\s+"," ",s).strip()
VOW="aăâeêioôơuưyàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ"
def exact(t): return re.compile(r"(?<![0-9a-zà-ỹ])"+r"\s+".join(map(re.escape,t.split()))+r"(?![0-9a-zà-ỹ])")
def fuzzy(t):
    out=[]
    for ch in t:
        if ch in VOW: out.append("["+VOW+"]+")
        elif ch==" ": out.append(r"\s+")
        else: out.append(re.escape(ch))
    return re.compile("".join(out))
texts={}
for f in sorted(glob.glob(REPORTS+"/NVB_*CN_BCTN.pdf")):
    k=re.search(r"NVB_(\d\d)",f).group(1); txt=subprocess.run(["pdftotext","-layout",f,"-"],capture_output=True,text=True).stdout; texts[2000+int(k)]=norm(txt)
rows=[]
for t,dim in dd.items():
    ce=cf=0; e=exact(t); fz=fuzzy(t)
    for y,x in texts.items(): ce+=len(e.findall(x)); cf+=len(fz.findall(x))
    rows.append(dict(term=t,dim=dim,n_tokens=len(t.split()),n_chars=len(t.replace(" ","")),exact=ce,fuzzy=cf))
D=pd.DataFrame(rows); D["fp"]=D.fuzzy-D.exact
print("words per report",{y:len(x.split()) for y,x in texts.items()})
print("TOTAL exact",D.exact.sum(),"fuzzy",D.fuzzy.sum(),"inflation x%.2f"%(D.fuzzy.sum()/D.exact.sum()))
for lab,m in [("multi-word",D.n_tokens>1),("single-token",D.n_tokens==1),("single-token <=3 chars",(D.n_tokens==1)&(D.n_chars<=3))]:
    print(lab,"terms",int(m.sum()),"exact",int(D[m].exact.sum()),"fuzzy",int(D[m].fuzzy.sum()))
print(D.sort_values("fp",ascending=False).head(12)[["term","exact","fuzzy"]].to_string(index=False))
D.to_csv("results/exact_vs_fuzzy_NCB.csv",index=False)
# generic single-token terms: share of exact hits and effect on index
per=[]
for y,x in texts.items():
    w=len(x.split()); a=b=0
    for t,dim in dd.items():
        c=len(exact(t).findall(x)); a+=c; b+= c if len(t.split())>1 else 0
    per.append(dict(year=y,words=w,S_all=a/w,S_multi=b/w,hits_all=a,hits_multi=b))
Pdf=pd.DataFrame(per); print(Pdf.round(4).to_string(index=False)); print("corr S_all vs S_multi-only:",round(Pdf.S_all.corr(Pdf.S_multi),3),"| share of hits from single-token terms: %.2f"%(1-Pdf.hits_multi.sum()/Pdf.hits_all.sum()))
Pdf.to_csv("results/ncb_generic_terms.csv",index=False)
top1=D[(D.n_tokens==1)].sort_values("exact",ascending=False).head(8)[["term","exact"]]; print(top1.to_string(index=False))
