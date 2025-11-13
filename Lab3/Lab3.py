from transformers import BertTokenizer, BertForMaskedLM
import torch

name = 'bert-base-multilingual-uncased'
tokenizer = BertTokenizer.from_pretrained(name)
model = BertForMaskedLM.from_pretrained(name, return_dict = True)

text = "Команда нашего офиса активно работала над заданием последние три месяца. Группе завершить разработку " + tokenizer.mask_token + " удалось, но, к сожалению, им не хватило времени."

inputs = tokenizer.encode_plus(text,return_tensors='pt')
mask_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]
outputs = model(**inputs)
logits = outputs.logits
softmax = torch.softmax(logits, dim=-1)
mask_word = softmax[0, mask_index,:]
top = torch.topk(mask_word, 10, dim=1)
tokens = [tokenizer.decode([t]) for t in top.indices[0]]

print("Предложение:", text)
print("Топ 10 предсказаний:", tokens)