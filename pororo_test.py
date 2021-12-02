import json
from pororo import Pororo
import traceback

with open('./text_summary_dataset_1125/test_summary.json', encoding='UTF-8') as file:
    test_dataset = json.load(file)

print("There are {} paragrphs in the test set.\n".format(len(test_dataset)))

print("The first paragraph in the test set: ")
print(json.dumps(test_dataset[0], indent='\t', ensure_ascii=False))

summ = Pororo(task="summarization", model="abstractive", lang="ko")

cnt = 0
my_summaries = []
error_index = []
for paragraph in test_dataset:
    cnt += 1
    print(cnt)
    original = paragraph['original']
    original = original.replace('.', '.\n')
    # pick 3 sentences randomly and update 'summary'

    if cnt > 5:
        break

    try:
        summary = summ(original, top_k=3, top_p=3)
        paragraph['summary'] = summary
        my_summaries.append(paragraph)
    except:
        print(traceback.format_exc())
        # original = original.replace(u"\u200b",'')
        summary = summ(original)
        paragraph['summary'] = summary
        my_summaries.append(paragraph)
        error_index.append(cnt)
        print('error')
        continue
    #print(summary)
    #print(paragraph)

print('error index')
print(len(error_index))
for i in error_index:
    print(i)

with open('my_summary_sol.json', 'w', encoding="UTF-8") as file:
    json.dump(my_summaries, file, indent='\t', ensure_ascii=False)
    