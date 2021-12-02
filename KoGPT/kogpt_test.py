import torch
from transformers import AutoTokenizer, AutoModelForCausalLM 

tokenizer = AutoTokenizer.from_pretrained(
  'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
  bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
)
model = AutoModelForCausalLM.from_pretrained(
  'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
  pad_token_id=tokenizer.eos_token_id,
  torch_dtype='auto', low_cpu_mem_usage=True
).to(device='cuda', non_blocking=True)
_ = model.eval()

prompt = '인간처럼 생각하고, 행동하는 \'지능\'을 통해 인류가 이제까지 풀지 못했던'
with torch.no_grad():
  tokens = tokenizer.encode(prompt, return_tensors='pt').to(device='cuda', non_blocking=True)
  gen_tokens = model.generate(tokens, do_sample=True, temperature=0.8, max_length=64)
  generated = tokenizer.batch_decode(gen_tokens)[0]
  
print(generated)  # print: 인간처럼 생각하고, 행동하는 '지능'을 통해 인류가 이제까지 풀지 못했던 문제의 해답을 찾을 수 있을 것이다. 과학기술이 고도로 발달한 21세기를 살아갈 우리 아이들에게 가장 필요한 것은 사고력 훈련이다. 사고력 훈련을 통해, 세상
