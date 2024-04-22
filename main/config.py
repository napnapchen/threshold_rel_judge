LEGAL_TEST_COLLECTIONS ={
    'TRDL19':{
        'qrel_path': '../data/2019qrels-pass.txt',
        'query_path': '../data/msmarco-test2019-queries.tsv',
        'corpus_path': '../data/msmarco-v1-passage',
        'topic_list': [
            '1114819'
        ]
    }
}

SAMPLE_RULES = {
    4: [
        [0, 1, 2, 3],
        [3, 2, 1, 0],
        [3, 3, 0, 0],
        [0, 0, 3, 3],
        [0, 3, 3, 0],
        [3, 0, 0, 3],
    ],
    8: [
        [0, 0, 1, 1, 2, 2, 3, 3],
        [3, 3, 2, 2, 1, 1, 0, 0],
        [0, 1, 2, 3, 3, 2, 1, 0],
        [0, 1, 2, 3, 0, 1, 2, 3],
        [3, 2, 1, 0, 3, 2, 1, 0],
        [3, 2, 1, 0, 0, 1, 2, 3],
        [1, 2, 0, 3, 3, 0, 2, 1],
        [1, 2, 0, 3, 2, 1, 3, 0],
        [2, 1, 3, 0, 1, 2, 0, 3],
    ]
}

PATH_API_KEY_FILE = '../.apikey'
