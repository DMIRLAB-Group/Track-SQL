import requests
import json

url = ""

def request_one(sentence1, sentence2):
    response = requests.post(url, json={"sentence1": sentence1,"sentence2": sentence2})
    response = json.loads(response.text)
    sentence1_embedding = response.get("sentence1_embedding", None)
    sentence2_embedding = response.get("sentence2_embedding", None)
    metric = response.get("metric", None)
    assert sentence1_embedding is not None and sentence2_embedding is not None

    return sentence1_embedding, sentence2_embedding, metric

if __name__ == '__main__':
    sentence1 = "abc"
    sentence2 = "efd"
    emb1, emb2, metric = request_one(sentence1, sentence2)
    print(emb1)
    print(emb2)
    print(metric)