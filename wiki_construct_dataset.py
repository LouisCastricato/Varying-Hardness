from bs4 import BeautifulSoup
from datasets import load_dataset
import json
import multiprocessing as mp
import sys
from tqdm import tqdm

mbs = 10

wiki_dpr = load_dataset('wiki_dpr', 'psgs_w100.nq.no_index', split='train')

# construct elasticsearch index
wiki_dpr.load_elasticsearch_index("context", host="localhost", port="9200", es_index_name="wiki_dpr_full_es")

# load natural questions
natural_questions = load_dataset("natural_questions", split="train")
elem_list = []


def compute_jsonl_line(elem_dict_out, pos, neg):
    answer = elem_dict_out['answer'].upper()

    # process positive articles
    positive_articles = pos['text']
    positive_articles_ids = pos['id']
    positive_articles_upper = list(map(lambda x: x.upper(), positive_articles))


    # process negative articles
    negative_articles = neg['text']
    negative_articles_ids = neg['id']
    negative_articles_upper = list(map(lambda x: x.upper(), negative_articles))


    elem_dict_out['hard_negative'] = []
    # determine which articles do NOT contain the answer
    for i in range(len(negative_articles_upper)):
        article_set = set(negative_articles_upper[i].split())
        answer_set = set(answer.split())
        length_del = len(article_set.union(answer_set) - article_set)
        if length_del != 0:
            elem_dict_out['hard_negative'].append(negative_articles_ids[i])


    elem_dict_out['positive'] = []
    # determine which articles do contain the answer
    for i in range(len(positive_articles_upper)):
        article_set = set(positive_articles_upper[i].split())
        answer_set = set(answer.split())
        length_del = len(article_set.union(answer_set) - article_set)
        if length_del == 0:
            elem_dict_out['positive'].append(positive_articles_ids[i])


    return elem_dict_out

def hard_negative_mine(idx):
    elem = natural_questions[idx*mbs:(idx+1)*mbs]
    # create the output dictionary for jsonl.
    elem_dict_list = []
    for idx in range(len(elem['id'])):
        q = elem['question'][idx]['text']
        a = elem['annotations'][idx]['short_answers'][0]['text']

        # if there is no answer, skips this question
        if len(a) == 0:
            continue
        else:
            a = a[0]
        elem_dict = {
            "question" : q,
            'answer' : a
        }    

        # if the answer is empty, skip
        if len(elem_dict['answer']) != 0:
            elem_dict_list.append(elem_dict)

    query_list = list(map(lambda x: x['question'], elem_dict_list))
    query_answer_list = list(map(lambda x: x['question'] + " " + x['answer'], elem_dict_list))

    _, positive_examples = wiki_dpr.get_nearest_examples_batch("context", query_answer_list, k=5)
    _, negative_examples = wiki_dpr.get_nearest_examples_batch("context", query_list, k=5)

    input = zip(elem_dict_list, positive_examples, negative_examples)    
    output = list(map(lambda x: compute_jsonl_line(x[0], x[1], x[2]), input))

    return output


# multithread the hard negative mining
pool = mp.Pool(12)
range_list = list(map(lambda x: -x, list(range(0, len(natural_questions), mbs))))
elem_list = list(tqdm(pool.imap(hard_negative_mine, range_list), total=len(range_list)))
pool.close()

print(elem_list[-1])

# filter out all ements of elem_list that are None
elem_list = [elem for elem in elem_list if elem is not None]

# save list of dictionaries to a jsonl file
with open('wiki_dpr_full_es.jsonl', 'w') as f:
    for elem in elem_list:
        f.write(json.dumps(elem) + '\n')

