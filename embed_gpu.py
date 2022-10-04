import sys
sys.path.insert(0, "/data/disk2/private/caishihuai/low_data_generation/retrieval_database/contriever")
import json
import torch
from transformers import AutoTokenizer
from src.contriever import Contriever

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

TOKENIZER = AutoTokenizer.from_pretrained('facebook/mcontriever')
MODEL = Contriever.from_pretrained('facebook/mcontriever')
MODEL.eval()
MODEL = MODEL.to(device)

print("model loaded.")

index = 0
ctr = 0
root = "results/gpu_keys/key_"
suffix = ".pt"
embeddings = []

with open("results/text_database_with_label.json", "r") as input_file:
    for line in input_file:
        data = json.loads(line)
        neighbor = data["neighbor"]
        input = TOKENIZER(neighbor, padding=True, truncation=True, return_tensors='pt')
        input = input.to(device)
        output = MODEL(**input)
        output = output.cpu()
        embeddings.append(output)
        ctr += 1
        if ctr == 250:
            filename = root + str(index) + suffix
            torch.save(embeddings, filename)
            embeddings.clear()
            index += 1 
            ctr = 0

if embeddings:
    filename = root + str(index) + suffix
    torch.save(embeddings, filename)
    embeddings.clear()
