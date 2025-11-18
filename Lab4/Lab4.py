from sys import argv
import numpy as np
import torch
from jinja2.compiler import generate
from transformers import GPT2LMHeadModel, GPT2Tokenizer

np.random.seed(42)
torch.manual_seed(42)

def load_tokenizer_and_model(model_name_or_path):
    return GPT2Tokenizer.from_pretrained(model_name_or_path), \
           GPT2LMHeadModel.from_pretrained(model_name_or_path)

def generate_text(model, tok, prompt):
    input_ids = tok.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=300,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3,
        repetition_penalty=5.0,
        temperature=1
    )
    return tok.decode(output[0], skip_special_tokens=True)

prompt = '''История взаимодействия человека с цифровыми системами прошла сложный путь. Изначально общение осуществлялось посредством перфокарт и ламповых индикаторов, требуя глубоких специальных знаний. Следующей революцией стала концепция прямого манипулирования объектами на экране с помощью курсора. Этот графический интерфейс, управляемый механическим манипулятором, кардинально изменил парадигму, сделав технологии доступными для масс.

Современный этап характеризуется стремлением к полной иммерсивности и естественности. Устройства ввода эволюционируют в сторону распознавания жестов, голоса и даже направления взгляда. Центром такой системы является мощное вычислительное ядро, способное в реальном времени обрабатывать массивы данных с камер и микрофонов, предугадывая намерения оператора. Фокус смещается с локального интерфейса на голографические дисплеи и повсеместные сенсоры, встроенные в окружающую среду.

Однако, несмотря на всю сложность архитектуры, для рядового пользователя весь этот технологический комплекс сводится к двум основным ролям в его доме: универсальному рабочему инструменту с интерактивным монитором и центральному развлекательному устройству с панелью для отображения видеоконтента. Первое мы привыкли называть в творительном падеже'''

tok, model = load_tokenizer_and_model("sberbank-ai/rugpt3large_based_on_gpt2")
generated = generate_text(model, tok, prompt)
print(generated)
