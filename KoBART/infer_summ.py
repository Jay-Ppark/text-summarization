import torch
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration
import json
from tqdm.auto import tqdm

def load_model():
    model = BartForConditionalGeneration.from_pretrained('./kobart_summary')
    # tokenizer = get_kobart_tokenizer()
    return model

with open('../text_summary_dataset_1125/test_summary.json', encoding='UTF-8') as file:
    test_dataset = json.load(file)

model = load_model()
tokenizer = get_kobart_tokenizer()

my_summaries = []
error_index = []
cnt = 0
for paragraph in tqdm(test_dataset):
    if cnt > 5:
        break
    cnt += 1
    original = paragraph['original']
    original = original.replace('.', '.\n')

    input_ids = tokenizer.encode(original)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0)
    output = model.generate(input_ids, eos_token_id=1, max_length=256, num_beams=5)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)

    paragraph['summary'] = summary
    my_summaries.append(paragraph)

print('error index')
print(len(error_index))
for i in error_index:
    print(i)

with open('my_summary_sol.json', 'w', encoding="UTF-8") as file:
    json.dump(my_summaries, file, indent='\t', ensure_ascii=False)
