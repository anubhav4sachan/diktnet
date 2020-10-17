import os
from argparse import ArgumentParser
import json
import copy
import shutil


def process_kb(in_kb):
    all_values = set({})
    items = in_kb.get('items', {})
    if items is None:
        items = {}
    for item in items:
        for key, value in item.items():
            all_values.add(value)
    return sorted(all_values, key=len)


def flatten_entities(in_entities_map):
    result = []
    for key, values_list in in_entities_map.items():
        for value in values_list:
            if isinstance(value, dict):
                result += map(lambda x: str(x).lower(), value.values())
            else:
                result.append(str(value).lower())
    return sorted(result, key=len, reverse=True)


def extract_entities(in_utterance, in_kb_entries):
    result = set([])
    for kb_entry in in_kb_entries:
        if kb_entry in in_utterance:
            in_utterance = in_utterance.replace(kb_entry, '__entity__')
            result.add(kb_entry)
    return result 


def extract_entities_from_dialog(in_dialog, in_entities):
    result = set([])
    for turn in in_dialog['dialogue']:
        result.update(extract_entities(turn['data']['utterance'], in_entities))
    return result


def kb_entry_contains_all_entities(in_kb_entry, in_entities):
    found = set([])
    kb_entry_str = json.dumps(in_kb_entry)
    for entity in in_entities:
        if entity in kb_entry_str:
            found.add(entity)
    return len(found) == len(in_entities)


def delexicalize_dialog(in_dialog, in_entities_list):
    result = copy.deepcopy(in_dialog)
    result['scenario']['kb'] = json.loads(json.dumps(result['scenario']['kb']).lower())
    for turn in result['dialogue']:
        turn['data']['utterance'] = turn['data']['utterance'].lower()
    dialog_entities = extract_entities_from_dialog(in_dialog, in_entities_list)
    if result['scenario']['kb']['items']:
        for entry in result['scenario']['kb']['items']:
            if kb_entry_contains_all_entities(entry, dialog_entities):
                result['scenario']['kb']['items'] = [entry]
                print('New kb: {}'.format(json.dumps(entry)))
                break
    return result


def process_dataset(in_dataset_folder):
    datasets = {}
    for dataset_name in ['train', 'dev', 'test']:
        filename = 'kvret_{}_public.json'.format(dataset_name)
        with open(os.path.join(in_dataset_folder, filename)) as dataset_in:
            datasets[filename] = json.load(dataset_in)
    with open(os.path.join(in_dataset_folder, 'kvret_entities.json')) as entities_in:
        entities = json.load(entities_in)
    entities_flat = flatten_entities(entities)

    for dataset_name, dataset in datasets.items():
        for idx, dialog in enumerate(dataset):
            dataset[idx] = delexicalize_dialog(dialog, entities_flat)
    return datasets


def save_dataset(in_src_folder, in_tgt_folder, in_datasets):
    if not os.path.exists(in_tgt_folder):
        os.makedirs(in_tgt_folder)
    for filename in os.listdir(in_src_folder):
        if filename not in in_datasets:
            if os.path.isdir(os.path.join(in_src_folder, filename)):
                shutil.copytree(os.path.join(in_src_folder, filename),
                                os.path.join(in_tgt_folder, filename))
            else:
                shutil.copy(os.path.join(in_src_folder, filename), in_tgt_folder)
        else:
            with open(os.path.join(in_tgt_folder, filename), 'w') as json_out:
                json.dump(in_datasets[filename], json_out)


def configure_argument_parser():
    parser = ArgumentParser()
    parser.add_argument('dataset_folder')
    parser.add_argument('output_folder')
    return parser


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()
    datasets = process_dataset(args.dataset_folder)
    save_dataset(args.dataset_folder, args.output_folder, datasets)

