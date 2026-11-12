import re, unicodedata, subprocess, glob, random, pandas as pd, openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
exec(open("src/exact_vs_fuzzy.py").read().split("texts={}")[0])   # reuse dictionary + normaliser
print(pd.Series(dd).value_counts().to_dict())
random.seed(7); recs=[]; texts={}
for f in sorted(glob.glob(REPORTS+"/NVB_*CN_BCTN.pdf")):
    y=2000+int(re.search(r"NVB_(\d\d)",f).group(1)); texts[y]=norm(subprocess.run(["pdftotext","-layout",f,"-"],capture_output=True,text=True).stdout)
for y,x in texts.items():
    for t,dim in dd.items():
        for m in exact(t).finditer(x):
            toks=x[:m.start()].split()[-18:]; after=x[m.end():].split()[:18]
            recs.append(dict(bank="NVB",year=y,dimension=dim,term=t,context=" ".join(toks)+" [["+m.group(0)+"]] "+" ".join(after)))
R=pd.DataFrame(recs); print(len(R))
# stratified: up to 3 matches per (dimension x term-type x year) cell, max ~150
R["single"]=R.term.str.split().str.len()==1
S=pd.concat([g.sample(min(len(g),30),random_state=7) for _,g in R.groupby(["dimension","single"])]).reset_index(drop=True)
S.insert(0,"id",range(1,len(S)+1))
for c in ("coder1_is_DT_disclosure (1/0)","coder2_is_DT_disclosure (1/0)","notes"): S[c]=""
# recall sample: random 150-token passages with matched terms marked
P=[]
for y,x in texts.items():
    toks=x.split()
    for k in range(6):
        s=random.randrange(0,max(1,len(toks)-150)); seg=" ".join(toks[s:s+150]); P.append(dict(bank="NVB",year=y,passage=seg))
P=pd.DataFrame(P).sample(40,random_state=3).reset_index(drop=True); P.insert(0,"id",range(1,len(P)+1))
P["n_dictionary_matches"]=[sum(len(exact(t).findall(p)) for t in dd) for p in P.passage]
for c in ("coder1_DT_mentions_missed_by_dictionary (count)","coder2_DT_mentions_missed_by_dictionary (count)","examples_of_missed_terms"): P[c]=""
with pd.ExcelWriter("coding/coding_sample_NCB_pilot.xlsx",engine="openpyxl") as w:
    pd.DataFrame({"Instructions":[
     "PILOT coding sample drawn from the 11 NCB (NVB) annual reports only; extend the same script to all 26 banks/years (see make_coding_sample logic in match.py/coding.py).",
     "Sheet 'precision': each row is one exact dictionary match with context ([[ ]] marks the match). Two coders independently code 1 if the match refers to digital transformation/technology in a banking context, 0 otherwise (e.g. 'ai' meaning 'who', 'lan' meaning 'spread').",
     "Sheet 'recall': random 150-token passages. Coders count digital-transformation mentions that the dictionary did NOT match and list examples.",
     "Report: precision overall and by dimension/term type, recall = matched/(matched+missed), Cohen's kappa between coders (coders blind to each other's labels), after training on 20 extra items.",
     "Matches are exact, word-bounded, on NFC-lowercased text; source = PDF text layer (the main pipeline uses Tesseract OCR text)."]}).to_excel(w,sheet_name="README",index=False)
    S.drop(columns=["single"],errors="ignore").to_excel(w,sheet_name="precision",index=False); P.to_excel(w,sheet_name="recall",index=False)
    for ws in w.book.worksheets:
        for col in ws.columns: ws.column_dimensions[col[0].column_letter].width=min(60,max(12,max(len(str(c.value or "")) for c in col[:30])*0.9))
        for row in ws.iter_rows(min_row=2):
            for c in row: c.alignment=Alignment(wrap_text=True,vertical="top")
        for c in ws[1]: c.font=Font(bold=True)
print(len(S),len(P))
