import openai
from openai import OpenAI
import os
import time
import random
from tqdm import tqdm
import re

from utils import *
from config import *

API_KEY = None
# 打开文件，读取 API 密钥
# 打开文件，读取 API 密钥的第一行
with open(PATH_API_KEY_FILE, 'r') as api_file:
    API_KEY = api_file.readline().strip()  # 使用 readline() 读取第一行，并用 strip() 去除空格和换行符

#print(API_KEY)  # 打印 API 密钥以确认其内容

ROLE_DESCRIPTION = '\
You are an expert assessor making TREC (Text Retrieval Conference) relevance judgments.\
Indicate to what degree the text in a passage is relevant to the question and classify the relevance score\
into four classes, using a digit from 0 to 3.\n\
3 - Perfectly relevant: The passage is dedicated to the query and contains the exact answer.\n\
2 - Highly relevant: The passage has some answer for the query, but the answer may be a bit unclear,\
or hidden amongst extraneous information.\n\
1 - Related: The passage seems related to the query but does not answer it.\n\
0 - Irrelevant: The passage has nothing to do with the query.\n\
'


def create_rel_judge_prompt(query, target_id, example_list):
    question_prompt = "Question:" + query + "\n"
    example_prompt = "Before you made your judgment, you were provided with some cases for example.\n"
    for idx, item in enumerate(example_list):
        docid = item['docid']
        qrel = item['qrel']
        doc_content = get_doc_content(docid)
        example_prompt += f"\
        <passage_{idx+1}>{doc_content}\n\
        <relevance_{idx+1}>{qrel}\n\
        "

    target_content = get_doc_content(target_id)
    example_prompt += f"<passage_target>{target_content}\n"
    example_prompt += "\n Now, let's finish the relevance judgement of the target document step by step. \
    First, for each example above, generate a very brief explanation starting with the format such as \
    <explain_1>, <explain_2> ..., explaining the reasons for obtaining the given relevance score from \
    the perspective of relevance between the example passage and the question.\
    Then, decide to what degree the target passage is relevant to the question in <explain_target>, \
    and classify the relevance score in <relevance_target> for <passage_target> according to the same criteria \
    with the examples.\n"

    return question_prompt + example_prompt


def get_gpt_judge_response(prompt_for_judge, gpt_model="gpt-3.5-turbo"):
    # Placeholder for the API call
    # Note: Replace the next line with your actual API call
    #print(ROLE_DESCRIPTION)
    os.environ['OPENAI_API_KEY'] = API_KEY
    client = OpenAI()
    response = None
    try:
        # Simulate API call to get response
        response = client.chat.completions.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": ROLE_DESCRIPTION},
                {"role": "user", "content": prompt_for_judge}
            ],
            temperature=0.3,
            max_tokens=2048,
            top_p=1
        )

        # Attempt to retrieve the message from the first choice
        message = response.choices[0].message.content
        #print(message)
        return message

    except Exception as e:
        # Handle failures by throwing an error and printing it
        print(f"Request Failed: {e}")
        raise


def judge_docs_by_topic(qrel_file_path, query_map, topic_id, sample_rule, gpt_model="gpt-3.5-turbo", result={}):
    # get the example list
    example_list = get_ground_truth_xy_list(qrel_file_path, topic_id, sample_rule)

    # Call get_doc_list_for_judge with the example list to get the list for judge
    list_for_judge = get_doc_list_for_judge(qrel_file_path, topic_id, example_list)
    filtered_list = [item for item in list_for_judge if item not in result]
    query = query_map[topic_id]
    for doc_id in tqdm(filtered_list):
        rel_judge_prompt = create_rel_judge_prompt(query, doc_id, example_list)
        try:
            response = get_gpt_judge_response(rel_judge_prompt, gpt_model)
            match = re.search(r"<relevance_target>\s*(\d+)", response)
            score = int(match.group(1)) if match else None
            result[doc_id] = {"score": score, "reason": response}
        except:
            print(f"Error. topic:{topic_id} doc: {doc_id}")
        # Random sleep between 1 and 3 seconds
        sleep_time = random.randint(514, 810)*0.001
        time.sleep(sleep_time)
    return result
