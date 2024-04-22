from config import *
from utils import *
from gpt_judge import *
import json
import os
import argparse


class AssessorGPT:
    def __init__(self, test_collection):
        # 初始化时存储 test_collection 参数
        if test_collection not in LEGAL_TEST_COLLECTIONS:
            raise ValueError(f"Invalid test collection: {test_collection}.")

        self.test_collection = test_collection
        self.qrel_path = LEGAL_TEST_COLLECTIONS[test_collection]['qrel_path']
        self.query_path = LEGAL_TEST_COLLECTIONS[test_collection]['query_path']
        self.corpus_path = LEGAL_TEST_COLLECTIONS[test_collection]['corpus_path']
        self.topic_list =  LEGAL_TEST_COLLECTIONS[test_collection]['topic_list']
        self.query_map = get_query_map(self.query_path)

    def run_assess(self, shot_num, gpt_model):
        if shot_num not in SAMPLE_RULES:
            raise ValueError(f"Invalid example list length: {shot_num}.")
        for topic_id in self.topic_list:
            print(f"Handling topic {topic_id}.")
            for sample_rule in SAMPLE_RULES[shot_num]:
                sample_rule_name = ''.join(str(num) for num in sample_rule)
                print(f"- Handling sample_rule {sample_rule_name}.")
                result_file_name = f"topic_{topic_id}_rule_{sample_rule_name}_model_{gpt_model}.json"
                result_file_path = f'./output/{result_file_name}'
                if os.path.exists(result_file_path):
                    with open(result_file_path, 'r') as file:
                        current_result = json.load(file)
                else:
                    current_result = {}
                try:
                    result = judge_docs_by_topic(self.qrel_path, self.query_map,
                                                 topic_id, sample_rule, gpt_model=gpt_model, result=current_result)
                    with open(result_file_path, 'w') as res_file:
                        json.dump(result, res_file)
                except Exception as e:
                    # Handle failures by throwing an error and printing it
                    print(f"Something went wrong in topic {topic_id}: {e}")


def main(args):
    assessor_gpt = AssessorGPT(args.test_collection)
    assessor_gpt.run_assess(args.example_length, args.gpt_model)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parsing command line arguments')
    parser.add_argument('test_collection', type=str, help='The name of test collection')
    parser.add_argument('-l', '--example_length', type=int, help='Length of example sequence', default=8)
    parser.add_argument('-m', '--gpt_model', type=str, help='GPT model to use as assessor', default='gpt-3.5-turbo')
    args = parser.parse_args()
    main(args)



