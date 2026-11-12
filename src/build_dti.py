"""Rebuild the DTI series from the OCR text reports with EXACT phrase matching and the 117-term dictionary.
Usage (repo root):  python src/build_dti.py <folder_with_txt_reports> data/panel_dti_financials_exact117.csv
Cleaning is the notebook's clean_text_pipeline (unchanged, so token counts equal those of the original panel);
matching is exact: the diacritic-preserving token sequence, flexible whitespace between tokens, no adjacent
alphanumeric characters. LnDTI = ln(1+F) as in the paper; W_DTI = sum_k tf_k*idf_k, idf_k = ln(N/(1+df_k))+1."""
import re, unicodedata, math, glob, sys, os
import pandas as pd
import re, unicodedata

def clean_text_pipeline(raw):
    # normalize unicode
    text = raw.replace('\r','\n')
    text = unicodedata.normalize('NFKC', text)
    lines = text.splitlines()
    cleaned_lines = []
    prev = ""
    for ln in lines:
        ln_strip = ln.strip()
        if ln_strip == "":
            prev = ln
            continue
        # skip lines that are numeric-only / page numbers / table headers
        if re.fullmatch(r'[\s\W\d\-]{1,10}', ln_strip):
            prev = ln
            continue
        low = ln_strip.lower()
        # skip common header/footer markers (tune this list if needed)
        if any(h in low for h in ['báo cáo', 'thường niên','chương','tổng quan','mục lục','mẫu số','thông tin','thuyết minh','báo cáo tài chính']):
            prev = ln
            continue
        # join hyphen-broken words
        if prev.endswith('-') and cleaned_lines:
            cleaned_lines[-1] = cleaned_lines[-1].rstrip('-') + ln_strip
        else:
            cleaned_lines.append(ln_strip)
        prev = ln
    t = " ".join(cleaned_lines)
    t = t.lower()
    # remove some nuisances
    t = re.sub(r'[®©•×¶§†■◆►…·“”‘’‚]', ' ', t)
    t = re.sub(r'[_\*\=\+\>\<\^\~\|\\\/]+', ' ', t)
    t = ''.join(ch for ch in t if ch.isprintable())
    t = re.sub(r'\s+', ' ', t).strip()
    return t

D=pd.read_csv("data/dictionary_120_terms.csv"); D=D[D.in_main_index]
DIMS=["DTI_UX","DTI_ST","DTI_PL","DTI_OP","DTI_CU","DTI_DA"]
def pat(t): return re.compile(r"(?<![^\W_])"+r"\s+".join(map(re.escape,t.split()))+r"(?![^\W_])",re.U)
P={k:pat(k) for k in D.keyword}; DIM=dict(zip(D.keyword,D.dimension))
def deacc(x):
    x=unicodedata.normalize("NFKD",x.replace("đ","d").replace("Đ","D")); return "".join(c for c in x if not unicodedata.combining(c))
PD={k:pat(deacc(k)) for k in D.keyword}
assert len({deacc(k) for k in D.keyword})==len(D), "two keywords collide after accent removal"
DIAC=set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
NODIAC=0.10   # reports whose OCR text keeps fewer than 10% accented letters are matched on accent-stripped text
def run(folder,out):
    files=sorted(glob.glob(os.path.join(folder,"**","*.txt"),recursive=True)); print(len(files),"reports")
    cnt={}; words={}; flag={}
    for f in files:
        raw=open(f,encoding="utf8",errors="ignore").read(); t=clean_text_pipeline(raw)
        words[os.path.basename(f)]=len(re.findall(r"\w+",t)); t2=re.sub(r"[\W_]+"," ",t)
        letters=re.findall(r"[a-zà-ỹđ]",t2); share=sum(c in DIAC for c in letters)/max(len(letters),1)
        nod=False; flag[os.path.basename(f)]=share<NODIAC   # flagged only: such OCR files are excluded from the analyses (see common.load)
        cnt[os.path.basename(f)]={k:len((PD if nod else P)[k].findall(deacc(t2) if nod else t2)) for k in P}
    N=len(files); dfk={k:sum(1 for d in cnt if cnt[d][k]>0) for k in P}
    idf={k:math.log(N/(1+dfk[k])+1e-12)+1.0 for k in P}
    rows=[]
    for d,c in cnt.items():
        F=sum(c.values()); w=words[d]
        r=dict(doc=d,ocr_no_diacritics=int(flag[d]),total_keyword_hits=F,total_words=w,S_DTI=F/w,LnDTI=math.log(1+F),W_DTI=sum(v*idf[k] for k,v in c.items()))
        for m in DIMS:
            f=sum(v for k,v in c.items() if DIM[k]==m)
            r[m+"_freq"]=f; r[m+"_S_DTI"]=f/w; r[m+"_LnDTI"]=math.log(1+f); r[m+"_W_DTI"]=sum(v*idf[k] for k,v in c.items() if DIM[k]==m)
        rows.append(r)
    R=pd.DataFrame(rows); R["_key"]=R.doc.str.replace(r"\.txt$","",regex=True).str.upper()
    old=pd.read_csv("data/panel_dti_financials.csv")[["_key","Bank","Year","ROE","ROA","TotalAssets","SIZE"]]
    R=R.merge(old,on="_key",how="inner"); R.to_csv(out,index=False); print(R.shape,"->",out)
    kw=pd.DataFrame({d:cnt[d] for d in cnt}); kw.index.name="keyword"; kw.to_csv(out.replace(".csv","_keyword_counts.csv"))
if __name__=="__main__": run(sys.argv[1],sys.argv[2])
