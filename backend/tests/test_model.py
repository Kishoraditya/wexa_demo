
import torch  
from transformers import AutoTokenizer, AutoModelForCausalLM  
from peft import PeftModel  
  
BASE = "microsoft/Phi-3-mini-4k-instruct"  
ADAPTER = "kishoraditya/enterprise-rag-adapter"  
  
print("Loading tokenizer...")  
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)  
  
print("Loading base model (float32, CPU)...")  
model = AutoModelForCausalLM.from_pretrained(  
    BASE,  
    device_map="cpu",  
    trust_remote_code=True,  
    torch_dtype=torch.float32,  
)  
  
print("Applying LoRA adapter...")  
model = PeftModel.from_pretrained(model, ADAPTER)  
model.eval()  
  
print("SUCCESS — running test inference...")  
inputs = tok("What is reliability?", return_tensors="pt")  
out = model.generate(**inputs, max_new_tokens=50)  
print(tok.decode(out[0], skip_special_tokens=True))  
