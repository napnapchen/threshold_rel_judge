import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.lucene import LuceneSearcher
import json

searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')


def get_doc_content(docid:str):
    doc = searcher.doc(docid)
    json_doc = json.loads(doc.raw())
    doc_content = json_doc['contents']
    return doc_content


def get_query_map(file_path):
    # Path to the file
    #file_path = '/mnt/data/msmarco-test2019-queries.tsv'

    # Initialize an empty dictionary to store the queries
    queries_dict = {}

    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Strip the newline character and split the line by the tab delimiter
            query_id, query = line.strip().split('\t')
            # Add the query_id and query to the dictionary
            queries_dict[query_id] = query

    # (Optional) Print the size of the dictionary and a few example entries
    print(f'Total queries parsed: {len(queries_dict)}')
    #for query_id, query in list(queries_dict.items())[:5]:
    #    print(f'{query_id}: {query}')
    return queries_dict


# Can be used for sampling example list.
def get_ground_truth_xy_list(file_path, topic_id, sample_rule):
    # Initialize a dictionary to store topic data
    topic_data = {}

    # Read the file and populate the dictionary
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            current_topic_id, _, doc_id, relevance_score = int(parts[0]), parts[1], parts[2], int(parts[3])

            # Only process entries for the specified topic_id
            if current_topic_id == topic_id or str(current_topic_id) == str(topic_id):
                if relevance_score not in topic_data:
                    topic_data[relevance_score] = []
                topic_data[relevance_score].append(doc_id)

    # If sample_rule is None or an empty list, return all doc_id - qrel pairs for the topic_id
    if sample_rule is None or not sample_rule:
        result_docs = []
        for score, docs in topic_data.items():
            for doc_id in docs:
                result_docs.append({"docid": doc_id, "qrel": score})
        return result_docs

    # Check if documents exist for each relevance score in sample_rule
    result_docs = []
    for score in sample_rule:
        if score in topic_data:
            # Take the first document ID that matches the score
            result_docs.append({"docid": topic_data[score][0], "qrel": score})
        else:
            raise ValueError(f"No documents found for topic {topic_id} with relevance score {score}")

    return result_docs


# Define a function to get a list of document IDs for a specific topic that are not in the example_list
def get_doc_list_for_judge(file_path, topic_id, example_list):
    # Initialize a list to store document IDs
    all_docs = set()
    example_id_list = [example['docid'] for example in example_list]
    # Read the file and collect all document IDs for the given topic_id
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            current_topic_id, _, doc_id, _ = int(parts[0]), parts[1], parts[2], int(parts[3])

            # If the current line's topic matches the specified topic_id, add the doc_id to the set
            if current_topic_id == topic_id or str(current_topic_id) == str(topic_id):
                all_docs.add(doc_id)

    # Filter out documents that are in the example_list
    filtered_docs = list(all_docs.difference(set(example_id_list)))

    return filtered_docs





def extract_scores(input_dict):
    """
    Extracts scores from the nested dictionaries and returns them in a new dictionary.

    Parameters:
    input_dict (dict): A dictionary where each value is another dictionary with 'score' and 'reason' keys.

    Returns:
    dict: A dictionary with the same keys as input_dict, but values are the 'score' from each nested dictionary.
    """
    output_dict = {}
    for key, value in input_dict.items():
        if 'score' in value:
            output_dict[key] = value['score']
    return output_dict


def compare_scores(predicted_scores, ground_truth_scores):
    """
    Compare the score distributions between predicted scores and ground_truth scores and return percentage distribution.

    :param predicted_scores: Dictionary of doc_id to predicted scores
    :param ground_truth_scores: Dictionary of doc_id to ground_truth scores
    :return: A dictionary that maps (predicted_score, ground_truth_score) tuples to their percentage counts
    """
    score_distribution = {}
    total_count = 0

    for doc_id, predicted_score in predicted_scores.items():
        if doc_id in ground_truth_scores:
            ground_truth_score = ground_truth_scores[doc_id]
            score_pair = (predicted_score, ground_truth_score)

            score_distribution[score_pair] = score_distribution.get(score_pair, 0) + 1
            total_count += 1

    # Convert counts to percentages
    if total_count > 0:
        for score_pair in score_distribution:
            score_distribution[score_pair] = (score_distribution[score_pair] / total_count) * 100

    return score_distribution



def plot_heatmap(score_distribution, title='Score Distribution Heatmap'):
    """
    Plot a heatmap for the score distribution comparison.

    :param score_distribution: Distribution of score pairs from compare_scores function
    :param title: Title of the plot
    """
    # Initialize a 4x4 matrix to store frequencies
    matrix = np.zeros((4, 4))

    # Populate the matrix with frequencies
    for (predicted_score, ground_truth_score), frequency in score_distribution.items():
        matrix[predicted_score][ground_truth_score] = frequency

    # Plotting
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(matrix, annot=False, fmt="d", cmap='Blues', cbar=True)
    ax.set_xlabel('Ground Truth Score')
    ax.set_ylabel('Predicted Score')
    ax.set_title(title)
    plt.show()
