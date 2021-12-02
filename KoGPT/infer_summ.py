import yaml
import torch
import torch.nn.functional as F
from train_ptuning import KoGPTConditionalGeneration
from utils import generate_next_token

# hparams_file = 'log/hparams.yaml'
# with open(hparams_file) as f:
#     hparams = yaml.load(f)

# inf = KoGPTConditionalGeneration.load_from_checkpoint('./log/KoGPT2_summary-last.ckpt', hparams=hparams)
inf = KoGPTConditionalGeneration.load_from_checkpoint('./logs/model_chp/epoch=01-val_loss=2.282.ckpt')

tokenizer = inf.tokenizer
SUMMARY = '<unused1>'
PTUNING = '<unused2>'
EOS = '</s>'

import json
from tqdm.auto import tqdm

with open('../text_summary_dataset_1125/test_summary.json', encoding='UTF-8') as file:
    test_dataset = json.load(file)


my_summaries = []
error_index = []
cnt = 0
for paragraph in tqdm(test_dataset):
    if cnt > 5:
        break
    cnt += 1
    original = paragraph['original']
    original = original.replace('.', '.\n')

    text = original.replace('\n', '')
    input_tokens = tokenizer.encode(PTUNING)* 10 + tokenizer.encode(text) + tokenizer.encode(SUMMARY)
    input_tensor = torch.tensor(input_tokens).unsqueeze(0)

    eos_id = tokenizer.encode(EOS)[0]

    while True:
        pred = inf.model(input_tensor)
        next_token = generate_next_token(pred.logits, temperature=1.0, top_p=0.8)

        if next_token.item() == eos_id:
            break
        else:
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)],1)

    # tokenizer.decode(input_tensor[0])

    summary = tokenizer.decode(input_tensor[0]).split('<unused1>')[-1].strip()

    paragraph['summary'] = summary
    my_summaries.append(paragraph)

print('error index')
print(len(error_index))
for i in error_index:
    print(i)

with open('my_summary_sol.json', 'w', encoding="UTF-8") as file:
    json.dump(my_summaries, file, indent='\t', ensure_ascii=False)
