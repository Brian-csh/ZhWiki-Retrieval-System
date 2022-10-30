import os
import sys
sys.path.insert(0, os.getcwd() + "/../contriever")
import json
import torch
from transformers import AutoTokenizer
from src.contriever import Contriever
from tqdm import tqdm
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="text file containing the neighbor-continuation pairs")
    parser.add_argument("output_dir", help="path to the output directory")
    parser.add_argument("tensors_per_file", type=int, help="number of tensors to be stored within a single .pt file")
    args = parser.parse_args()


    TOKENIZER = AutoTokenizer.from_pretrained('../mcontriever')
    MODEL = Contriever.from_pretrained('../mcontriever').to(device)
    MODEL.eval()
    print("model loaded.")

    index = 0
    ctr = 0
    root = args.output_dir + "/key_"
    suffix = ".pt"
    embeddings = []

    with open(args.input_file, "r") as input_file:
        for line in tqdm(input_file):
            data = json.loads(line)
            neighbor = data["neighbor"]
            input = TOKENIZER(neighbor, padding=True, truncation=True, return_tensors='pt').to(device)
            output = MODEL(**input)
            embeddings.append(output.cpu())
            ctr += 1
            if ctr == args.tensors_per_file:
                filename = root + str(index) + suffix
                torch.save(embeddings, filename)
                embeddings.clear()
                index += 1 
                ctr = 0

    if embeddings:
        filename = root + str(index) + suffix
        torch.save(embeddings, filename)
        embeddings.clear()

if __name__ == '__main__':
    main()