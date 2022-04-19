import json
import random
import re
from typing import Dict, List, Tuple

from tqdm import tqdm

from lib.query import Query
from lib.table import Table
import os


def get_query(query_dict: Dict, table_data: List[Dict]) -> Tuple[str, int]:
    query = Query.from_dict(query_dict['sql'])
    table_dict = next((line for line in table_data if line['id'] == query_dict['table_id']), None)

    if 'caption' not in table_dict:
        table_dict['caption'] = None
    table = Table(table_dict['id'], table_dict['header'], table_dict['types'], table_dict['rows'],
                  table_dict['caption'])

    query_str = table.query_str(query)

    return query_str.replace(Table.get_id(table_dict['id']), table_dict['custom_name']), table_data.index(table_dict)


if __name__ == "__main__":
    for dataset in ['train', 'test', 'dev']:

        query_data = []
        with open(f'data/{dataset}.jsonl') as f:
            for line in f:
                query_data.append(json.loads(line))

        table_data = []
        with open(f'data/{dataset}.tables.jsonl') as f:
            for line in f:
                table_data.append(json.loads(line))


        table_names = []
        table_strings = []
        for i, table in tqdm(enumerate(table_data)):
            table_name = ""

            if "caption" in table and table['caption']:
                table_name = table['caption']
            elif "name" in table and table['name']:
                table_name = table['name']
            elif 'page_title' in table and table['page_title']:
                table_name = table['page_title']
            elif 'section_title' in table and table['section_title']:
                table_name = table['section_title']
            else:
                table_name = table['id']

            table_name = table_name.replace(" ", "_")
            table_names.append(table_name)
            header_text = ", ".join([column_name.replace(" ", "_") for column_name in table['header']])
            # print(header_text)
            table_strings.append(f"# {table_name} ({header_text})")
            table_data[i]['custom_name'] = table_name

        # print(table_names)
        # print('\n'.join(table_strings))
        # print(len(table_data))
        # print(len(table_strings))

        jsonl_results = []

        for query_dict in tqdm(query_data):
            query, table_idx = get_query(query_dict, table_data)
            question = query_dict['question']
            table_str = table_strings[table_idx]
            selected_table_strs = random.sample(table_strings, random.randint(2, 4)) + [table_str]

            random.shuffle(selected_table_strs)

            preamble = ["### MySQL tables, with their properties:"
                        "#"]

            posterior = ["#",
                         f"# A query to answer the question: {question}", "###\\n"]

            all_lines = preamble + selected_table_strs + posterior

            prompt = '\\n'.join(all_lines)
            response = query + ";"

            if len(prompt) + len(response) <= 2048:
                jsonl_results.append(f'{{"prompt": "{prompt}", "response": "{response}"}}')
            else:
                print("Dropped")
                # print(response)

        jsonl_text = "\n".join(jsonl_results)

        jsonl_text = re.sub('\n(?!{)', '', jsonl_text)

        with open(f"gpt_{dataset}.jsonl", 'w') as file:
            file.write(jsonl_text)

    os.system('cat gpt_train.jsonl gpt_test.jsonl gpt_dev.jsonl > gpt_all.jsonl')  # lazy but works
