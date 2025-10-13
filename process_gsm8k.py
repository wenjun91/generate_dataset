mport json
from transformers import AutoTokenizer

# 以BERT为例（其他模型只需替换模型名称）
tokenizer = AutoTokenizer.from_pretrained("/mnt/l00919884/weight/deepseek_r1_w8a8_mtp")
batch_size = 1800
input_len = 2048
dataset = []
dataset_path = "./GSM8K.jsonl"
with open(dataset_path, 'r', encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        dataset.append(data['question'])
# repeat input_len
dataset_2k = []                                   
for sentence in dataset:
    words = tokenizer.tokenize(sentence)
    if len(words) == 0:
         continue
     len_num = len(words) // input_len
     if len_num == 0:
        multiplier = (input_len // len(words)) + 1
        repeated_len = words * multiplier
        words = repeated_len[:input_len]
        decoded_text = tokenizer.convert_tokens_to_string(words)
        dataset_2k.append(decoded_text)
# repeat to batch_size
batch_num = len(dataset_2k) // batch_size
if batch_num == 0:
    multiplier = (batch_size // len(dataset_2k)) + 1
    repeated_batch = dataset_2k * multiplier
    dataset_2k = repeated_batch[:batch_size]
else:
    dataset_2k = dataset_2k[:batch_size]

json_str = json.dumps(dataset_2k, ensure_ascii=False, indent=4)
print("gen start.........")
with open(f'GSM8K-in{input_len}-bs{batch_size}.jsonl', 'w', encoding='utf-8') as f:
    print("open file start.........")
    for i in range(len(dataset_2k)):
        f.write(json.dumps({"question": dataset_2k[i], "answer": "none"}, ensure_ascii=False))
        f.write("\n")
