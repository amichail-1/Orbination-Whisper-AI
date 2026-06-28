"""Q3_K-MATCHED QAT: the fake-quant calls the EXACT ggml Q3_K (ctypes) with STE,
so training == deployment. Export to standard whisper.cpp Q3_K keeps the quality."""
import json, time, os, re, statistics as st, sys, ctypes
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, librosa
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_cosine_schedule_with_warmup

LIB=ctypes.CDLL("/home/amichail/q3kqat/libq3k.so")
LIB.q3k_rt.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_long,ctypes.c_long]
def q3k_roundtrip(w):                      # w: 2D tensor [out,in], in%256==0
    a=np.ascontiguousarray(w.detach().to("cpu",torch.float32).numpy())
    out=np.empty_like(a)
    LIB.q3k_rt(a.ctypes.data_as(ctypes.c_void_p),out.ctypes.data_as(ctypes.c_void_p),a.shape[0],a.shape[1])
    return torch.from_numpy(out).to(w.device,w.dtype)
class Q3K(torch.autograd.Function):
    @staticmethod
    def forward(ctx,w): return q3k_roundtrip(w)
    @staticmethod
    def backward(ctx,g): return g          # straight-through
REFRESH_EVERY=int(os.environ.get("REFRESH","4"))   # recompute exact Q3_K every K steps (speed)
GSTEP=[0]
class Q3KLinear(nn.Module):
    def __init__(s,lin):
        super().__init__()
        s.weight=nn.Parameter(lin.weight.detach().clone())
        s.bias=nn.Parameter(lin.bias.detach().clone()) if lin.bias is not None else None
        s.cached=None                       # eval: quantize once, reuse across generation steps
        s.qc=None; s.last=-10**9            # train: lazy-refreshed quantized weight
    def forward(s,x):
        if s.training:
            if s.qc is None or (GSTEP[0]-s.last)>=REFRESH_EVERY:
                s.qc=q3k_roundtrip(s.weight).detach(); s.last=GSTEP[0]
            w=s.weight+(s.qc-s.weight).detach()      # STE with (lazily) exact Q3_K
        elif s.cached is not None:
            w=s.cached
        else:
            w=q3k_roundtrip(s.weight)
        return F.linear(x,w.to(x.dtype),None if s.bias is None else s.bias.to(x.dtype))

def set_eval_cache(model,on):
    for m in model.modules():
        if isinstance(m,Q3KLinear):
            m.cached=q3k_roundtrip(m.weight) if on else None
def wrap(model):
    n=0
    for name,mod in list(model.named_modules()):
        if not isinstance(mod,nn.Linear): continue
        if not (("encoder" in name) or ("decoder" in name)): continue
        leaf=name.rsplit(".",1)[-1]
        if leaf not in ("q_proj","k_proj","v_proj","out_proj","fc1","fc2"): continue
        if mod.in_features%256: continue
        parent=model.get_submodule(name.rsplit(".",1)[0])
        setattr(parent,leaf,Q3KLinear(mod)); n+=1
    return n

MAX_SECONDS=float(os.environ.get("MAX_SECONDS","2400"))
SEED="/home/amichail/runs/ml_ft/latest"; OUT="/home/amichail/q3kqat/run_q3kmatched"; os.makedirs(OUT,exist_ok=True)
LC={"Greek":"greek","Spanish":"spanish","French":"french","English":"english"}
dev,dt="cuda",torch.bfloat16
proc=WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
model=WhisperForConditionalGeneration.from_pretrained(SEED,torch_dtype=dt).to(dev)
model.config.forced_decoder_ids=None; model.config.use_cache=False
nq=wrap(model); model.to(dev)
print(f"[q3kmatched] wrapped {nq} linears with EXACT ggml Q3_K fake-quant (enc+dec)",flush=True)
teacher=WhisperForConditionalGeneration.from_pretrained(SEED,torch_dtype=dt).to(dev).eval()
teacher.config.use_cache=False
for p in teacher.parameters(): p.requires_grad_(False)

train=[json.loads(l) for l in open("/home/amichail/ml_train.jsonl")]
val=[json.loads(l) for l in open("/home/amichail/ml_val.jsonl")]
bylang={}
for r in val: bylang.setdefault(r["lang"],[]).append(r)
class DS(Dataset):
    def __len__(s): return len(train)
    def __getitem__(s,i): return train[i]
def collate(b):
    ys=[librosa.load(r["audio"],sr=16000,mono=True)[0] for r in b]
    feats=proc(ys,sampling_rate=16000,return_tensors="pt",padding="max_length",truncation=True).input_features
    lab=[]
    for r in b:
        proc.tokenizer.set_prefix_tokens(language=LC[r["lang"]],task="transcribe")
        lab.append(proc.tokenizer(r["sentence"],truncation=True,max_length=200).input_ids)
    m=max(len(x) for x in lab); labels=torch.full((len(b),m),-100,dtype=torch.long)
    for i,ids in enumerate(lab): labels[i,:len(ids)]=torch.tensor(ids)
    return feats,labels
loader=DataLoader(DS(),batch_size=8,shuffle=True,num_workers=6,collate_fn=collate,pin_memory=True,drop_last=True)

def norm(t): t=t.lower().strip(); t=re.sub(r"[\.,;:!\?\"'`´«»“”‘’\(\)\[\]/]"," ",t); return re.sub(r"\s+"," ",t).strip()
def wer(p,r):
    p=norm(p).split(); r=norm(r).split()
    if not r: return 0.0
    prev=list(range(len(p)+1))
    for i,rw in enumerate(r,1):
        cur=[i]+[0]*len(p)
        for j,pw in enumerate(p,1): cur[j]=min(prev[j]+1,cur[j-1]+1,prev[j-1]+(rw!=pw))
        prev=cur
    return prev[-1]/len(r)
@torch.no_grad()
def evaluate(step):
    model.eval(); set_eval_cache(model,True); res={}
    for lang,items in bylang.items():
        sub=items[:40]; ws=[]
        for i in range(0,len(sub),8):
            bb=sub[i:i+8]; ys=[librosa.load(r["audio"],sr=16000,mono=True)[0] for r in bb]
            f=proc(ys,sampling_rate=16000,return_tensors="pt",padding=True).input_features.to(dev,dt)
            ids=model.generate(f,language=LC[lang],task="transcribe",max_new_tokens=200,num_beams=5)
            for r,pr in zip(bb,proc.batch_decode(ids,skip_special_tokens=True)): ws.append(wer(pr,r["sentence"]))
        res[lang]=round(st.mean(ws),3)
    set_eval_cache(model,False); model.train()
    open(f"{OUT}/eval.jsonl","a").write(json.dumps({"step":step,"wer":res})+"\n")
    print(f"[eval] step {step} (EXACT Q3_K, beam5) {res}",flush=True)

optim=AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=0.0)
sched=get_cosine_schedule_with_warmup(optim,60,6000)
t0=time.time(); step=0; accum=1; T=2.0
print("[q3kmatched] eval BEFORE QAT (= what stock Q3_K export would give):",flush=True); evaluate(0)
model.train(); optim.zero_grad()
while time.time()-t0<MAX_SECONDS:
    for feats,labels in loader:
        if time.time()-t0>=MAX_SECONDS: break
        feats=feats.to(dev,dt); labels=labels.to(dev)
        out=model(input_features=feats,labels=labels); ce=out.loss
        with torch.no_grad(): tlog=teacher(input_features=feats,labels=labels).logits
        mask=labels.ne(-100); s=out.logits[mask]/T; tt=tlog[mask]/T
        kd=F.kl_div(F.log_softmax(s,-1),F.softmax(tt,-1),reduction="batchmean")*(T*T)
        loss=(0.5*ce+0.5*kd)/accum; loss.backward()
        if (step+1)%accum==0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
            optim.step(); sched.step(); optim.zero_grad()
        step+=1; GSTEP[0]=step
        if step%20==0: print(f"[q3kmatched] step {step} t={time.time()-t0:.0f}s ce={float(ce):.3f} kd={float(kd):.3f}",flush=True)
        if step%200==0: evaluate(step)
evaluate(step)
model.save_pretrained(OUT,safe_serialization=True); proc.save_pretrained(OUT)
print(f"[q3kmatched] DONE {step} steps -> {OUT}",flush=True)
