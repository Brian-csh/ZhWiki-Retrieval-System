import json

chunk_count = 0


# cut a chunk into neighbor-continuation pair
def process_chunk(chunk, article_title, article_id):
    global chunk_count
    data = {}
    if len(chunk) <= 64:
        neighbor = chunk[0:]
        continuation = ""
    else:
        neighbor = chunk[0:64]
        continuation = chunk[64:]

    data["neighbor"] = neighbor
    data["continuation"] = continuation
    data["index"] = chunk_count
    data["article_title"] = article_title
    data["article_id"] = article_id
    
    chunk_count += 1
    return data


# process the text of a wiki-article into neighbor-continuation pairs
def process_article_text(article_text, article_title, article_id):
    dataset = []
    while len(article_text) > 128:
        chunk = article_text[0:128]
        dataset.append(process_chunk(chunk, article_title, article_id))
        article_text = article_text[128:]

    if article_text != "":
        dataset.append(process_chunk(article_text, article_title, article_id))
    
    return dataset


# parse articles into the text database to be used for retrieval
def parse_article(article_json_string):
    database = []

    article_json = json.loads(article_json_string)
        
    article_title = article_json["title"]
    article_id = article_json["id"]

    article_text = ""

    # process the abstract
    abstract_json = article_json["meta"]["abstract"]
    
    for chunk in abstract_json:
        article_text += chunk
        
    # process the rest of the contents
    content_json = article_json["content"]
    for sub_content in content_json:
        sub_content_list = sub_content["sub_content"]
        for chunk in sub_content_list:
            article_text += chunk
    
    database.extend(process_article_text(article_text, article_title, article_id))

    return database


def main():
    database = []
    with open("wiki.json", "r") as input_file:
        for article in input_file:
            database.extend(parse_article(article))
    # articles = read_json_file("wiki.json")
    # database = parse_articles(articles)

    output_file = open("text_database.json", "w")
    for entry in database:
        json.dump(entry, output_file, ensure_ascii=False)
        output_file.write('\n')
    output_file.close()


if __name__ == "__main__":
    main()