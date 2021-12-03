import yaml
import torch
import torch.nn.functional as F
from train_ptuning import KoGPTConditionalGeneration
from utils import generate_next_token
from transformers import AutoTokenizer, AutoModelForCausalLM 
import json
from tqdm.auto import tqdm

with open('./data/test_summary.json', encoding='UTF-8') as file:
    test_dataset = json.load(file)

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

my_summaries = []
error_index = []
cnt = 0

for paragraph in tqdm(test_dataset):
    if cnt > 20:
        break
    cnt += 1
    original = paragraph['original']

    original = original.replace('.', '.\n')
    original = f'원본: {original}\n요약: '

    prompt = original
    with torch.no_grad():
        tokens = tokenizer.encode(prompt, return_tensors='pt').to(device='cuda', non_blocking=True)
        gen_tokens = model.generate(tokens, do_sample=True, temperature=0.8, max_length=1000, length_penalty=0.8, repetition_penalty=1.2)
        summary = tokenizer.batch_decode(gen_tokens)[0]

    # summary = summary.split('요약: ')[-1]
    paragraph['summary'] = summary
    my_summaries.append(paragraph)

with open('my_summary_sol.json', 'w', encoding="UTF-8") as file:
    json.dump(my_summaries, file, indent='\t', ensure_ascii=False)
