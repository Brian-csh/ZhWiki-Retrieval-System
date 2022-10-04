import json

chunk_count = 0

# returns a list consisting all the articles
def read_json_file(filename):
    json_file = open(filename, "r")
    articles = json_file.readlines()
    json_file.close()
    return articles


# chunk must be less than 128
def process_article_subchunk(article_subchunk, article_title, article_id):
    global chunk_count
    data = {}
    if len(article_subchunk) <= 64:
        neighbor = article_subchunk[0:]
        continuation = ""
    else:
        neighbor = article_subchunk[0:64]
        continuation = article_subchunk[64:]

    data["neighbor"] = neighbor
    data["continuation"] = continuation
    data["index"] = chunk_count
    data["article_title"] = article_title
    data["article_id"] = article_id
    
    chunk_count += 1
    return data


# process a chunk from the source
# return a dict consisiting of key, neighbor, and continuation
def process_article_chunk(article_chunk, article_title, article_id):
    dataset = []
    while len(article_chunk) > 128:
        subchunk = article_chunk[0:128]
        dataset.append(process_article_subchunk(subchunk, article_title, article_id))
        article_chunk = article_chunk[128:]

    dataset.append(process_article_subchunk(article_chunk, article_title, article_id))
    
    return dataset


# parse articles into data for the retrieval database
def parse_articles(articles):
    database = []
    for lines in articles:
        article = json.loads(lines)
        abstract = article["meta"]["abstract"]
        article_title = article["title"]
        article_id = article["id"]
        for article_chunk in abstract:
            database.extend(process_article_chunk(article_chunk, article_title, article_id))
        
        contents = article["content"]
        for content in contents:
            sub_content = content["sub_content"]
            for article_chunk in sub_content:
                database.extend(process_article_chunk(article_chunk, article_title, article_id))

    
    return database


articles = read_json_file("wiki.json") # specify file name
database = parse_articles(articles)

output_file = open("results/text_database_with_article_label.json", "w")
for entry in database:
    json.dump(entry, output_file, ensure_ascii=False)
    output_file.write('\n')
output_file.close()