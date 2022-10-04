import os
import sys
sys.path.insert(0, os.getcwd() + "/contriever")
import json
import math
import torch
import linecache
from tqdm import tqdm
from transformers import AutoTokenizer
from src.contriever import Contriever
import argparse

TOKENIZER = AutoTokenizer.from_pretrained('mcontriever')
print("tokenizer loaded")
MODEL = Contriever.from_pretrained('mcontriever')
print("model loaded")

device = device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


# computes L2 distance of two tensors
def l2_distance(a, b):
    dist = torch.square(a - b)
    dist = torch.sum(dist)
    return math.sqrt(dist)


# retrieves embeddings from pt file
def get_batch_embeddings(file_index):
    filename = "database/keys/key_" + str(file_index) + ".pt" # to change
    return torch.load(filename)


# update and maintain the knn list
def update_nn(nn_list, candidate, k):
    if len(nn_list) < k:
        nn_list.append(candidate)
        return
    for nn in nn_list:
        if candidate["dist"] < nn["dist"]:
            nn_list.remove(nn)
            nn_list.append(candidate)
            return


# retrieve text data given the nn_list which contains the index of the chunks
def retrieve_texts(nn_list):
    line_numbers = [nn["index"] for nn in sorted(nn_list, key=lambda d:d["dist"])]
    results = []
    for line in line_numbers:
        results.append(json.loads(linecache.getline("database/text_database.json", line + 1)))
    return results


# returns k nearest neighbors of sentence
def knn(sentence, k):
    input = TOKENIZER(sentence, padding=True, truncation=True, return_tensors='pt')
    query = MODEL(**input)
    query = query.to(device)
    nn_list = []
    for i in tqdm(range(39895)):
        embeddings = get_batch_embeddings(i)
        for idx, embedding in enumerate(embeddings):
            embedding = embedding.to(device)
            curr = {}
            curr["index"] = i * 500 + idx # to change
            curr["dist"] = l2_distance(query, embedding)
            update_nn(nn_list, curr, k)
    neighbors = retrieve_texts(nn_list)
    print(json.dumps(neighbors, ensure_ascii=False))
    return neighbors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="the chunks relevant to these query texts will be returned")
    parser.add_argument("k", help="k-nearest neighbors", type=int)
    args = parser.parse_args()

    knn(args.query, args.k)
