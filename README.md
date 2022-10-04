# ZhWiki Retrieval System
This repository is part of a research on low-data/few-shot text generation.
## Task Description
The ZhWiki Retrieval System searches relevant texts from the Chinese Wikipedia database, given some query in the form of text. The retrieval system uses the mcontriever model to pre-calculate the embeddings of the chunks from Chinese Wikipedia articles. To search for relevant chunks, the input query is embedded. Then, k-nearest neighbors (k-NN) algorithm is used to retrieve these chunks of text. The L2 distance is used to compute the similarity between the embeddings.
## Usage
### setup
- download [contriever] (https://github.com/facebookresearch/contriever) libary, and place it in the root directory of this project
- the model files should be placed in the folder `mcontriever`
- the embeddings are placed under keys folder, and the text database is in the file text_database.json. All of these are placed under a folder named `database`
```
    python knn.py <query> <k>
    # e.g. python knn.py "数学的历史中，欧几里得的著作" 8
```