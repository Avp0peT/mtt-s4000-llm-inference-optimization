"""7B V3 v3_0：prefill 分布逐位相同(0.0)但 token 序列分岔。
确认分岔步是否为近平局（若是，与实现无关）。"""
import sys
sys.path.insert(0, "src"); sys.path.insert(0, "benchmark"); sys.path.insert(0, "scripts")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ascend_opt import Request
from real_backend import TransformerDriver, SyntheticPromptSource, ForwardItem

MODEL="Qwen/Qwen2.5-7B-Instruct"; dev="musa:0"; GEN=16
tok=AutoTokenizer.from_pretrained(MODEL)
m=AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev).eval()
excl=[v for n in ("eos_token_id","pad_token_id","bos_token_id","unk_token_id")
      if isinstance((v:=getattr(tok,n,None)),int)]
src=SyntheticPromptSource(vocab_size=int(m.config.vocab_size),seed=1234,excluded_ids=excl)
r=Request(req_id="v3_0",prompt_len=16,gen_len=GEN)
ids=src.prompt_ids(r)

class G:
    def __init__(self,i): self._i=i; self.lg=None
    def __call__(self,*a,**k):
        o=self._i(*a,**k); self.lg=o.logits.detach().float(); return o
    def __getattr__(self,n): return getattr(self._i,n)

# 驱动逐 token 跑，每步记 top1/top2 gap
g=G(m); d=TransformerDriver(model_name=MODEL,device=dev,precision="fp16",
                            model=g,tokenizer=tok,prompt_source=src)
d.load()
o=d.forward([ForwardItem(req=r,n_new_tokens=16,need_token=True)])
print("驱动 prefill 末 top1/top2 gap:")
v,i=torch.topk(g.lg[0,-1,:],2); print(f"  top1={int(i[0])} top2={int(i[1])} gap={float(v[0]-v[1]):.4f}")
seq=[s.token_id for s in o.slots if s.token_id is not None]
for st in range(GEN-1):
    o=d.forward([ForwardItem(req=r,n_new_tokens=1,need_token=True)])
    v,i=torch.topk(g.lg[0,-1,:],2)
    t=[s.token_id for s in o.slots if s.token_id is not None]
    print(f"  decode{st}: tok={t} top1={int(i[0])} top2={int(i[1])} gap={float(v[0]-v[1]):.4f}")
    seq+=t
d.retire(["v3_0"])
print("drv seq:", seq)

# 参考侧逐步 gap
print("\n参考侧逐步 gap:")
past=None; cur=torch.tensor([ids],dtype=torch.long,device=dev); rseq=[]
with torch.no_grad():
    for st in range(GEN):
        out=m(input_ids=cur,past_key_values=past,use_cache=True); past=out.past_key_values
        lg=out.logits[0,-1,:].float(); v,i=torch.topk(lg,2)
        rseq.append(int(i[0]))
        print(f"  step{st}: tok={int(i[0])} top2={int(i[1])} gap={float(v[0]-v[1]):.4f}")
        cur=torch.tensor([[int(i[0])]],dtype=torch.long,device=dev)
print("ref seq:", rseq)
print("\n分岔位置:", next((k for k,(a,b) in enumerate(zip(seq,rseq)) if a!=b), None))
