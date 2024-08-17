import torch
import pandas as pd
import xml.etree.ElementTree as ET
from nltk.corpus import wordnet as wn
from collections import namedtuple

def load_xml_data(xml_file=''):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    data = []
    for text in root.findall('text'):
        for sentence in text.findall('sentence'):
            sentence_id = sentence.get('id')
            sentence_text = ' '.join([element.text for element in sentence])
            for instance in sentence.findall('instance'):
                instance_id = instance.get('id')
                lemma = instance.get('lemma')
                pos = instance.get('pos')
                word = instance.text
                data.append([sentence_id, instance_id, lemma, pos, word, sentence_text])
    columns = ['sentence_id', 'instance_id', 'lemma', 'pos', 'word', 'sentence_text']
    xml_data = pd.DataFrame(data, columns=columns)
    return xml_data


def load_gold_keys(gold_key_file=''):
    gold_key_data = []
    with open(gold_key_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            instance_id = parts[0]
            sense_id = parts[1]
            gold_key_data.append([instance_id, sense_id])

    # Create a DataFrame
    gold_key_columns = ['instance_id', 'sense_id']
    gold_key_df = pd.DataFrame(gold_key_data, columns=gold_key_columns)
    return gold_key_df


def format_sense_id(sense_id, cache):
    if sense_id not in cache:
        cache[sense_id] = wn.lemma_from_key(sense_id).synset().name()
    return cache[sense_id]


def load_balls(ball_file=''):
    Ball = namedtuple('Ball', ['center', 'distance', 'radius'])
    print("loading balls....")
    bdic = dict()
    with open(ball_file, 'r') as w2v:
        for line in w2v.readlines():
            wlst = line.strip().split()
            center = list(map(float, wlst[1:-2]))
            distance = float(wlst[-2])
            radius = float(wlst[-1])
            # Store the information in the dictionary using the named tuple
            bdic[wlst[0]] = Ball(center=center, distance=distance, radius=radius)
    print(len(bdic), 'balls are loaded\n')
    return bdic

def get_eval_pair(sense_index):
        prefix_groups = {}
        for sense, index in sense_index.items():
            prefix = sense.split('.')[0]
            if prefix not in prefix_groups:
                prefix_groups[prefix] = []
            prefix_groups[prefix].append(index)

        lemma_pair = {}
        lemmas = list(prefix_groups.keys())
        for idx, lemma in enumerate(lemmas):
            lemma_pair[idx] = prefix_groups[lemma]
            
        eval_pair = {}
        for sense, index in sense_index.items():
            prefix = sense.split('.')[0]
            eval_pair[index] = prefix_groups[prefix]
        
        return lemmas, lemma_pair, eval_pair

def preprocess_data(data_path, nball):
    # Load xml, gold key and nball
    xml_data = load_xml_data(data_path["xml"])
    gold_keys = load_gold_keys(data_path["gold_key"])
    # Merge data
    data_merged = pd.merge(xml_data, gold_keys, on='instance_id', how='inner')
    # Format sense ID and use the cache to store results
    sense_id_cache = {}
    data_merged['formatted_sense_id'] = data_merged['sense_id'].apply(lambda x: format_sense_id(x, sense_id_cache))
    data_merged = data_merged[data_merged['formatted_sense_id'].isin(nball.keys())]
    data_merged.reset_index(drop=True, inplace=True)

    # Keep it with nball
    sense_labels = list(nball.keys())
    sense_index = {sense: idx for idx, sense in enumerate(sense_labels)}
    lemma_labels, lemma_pair, eval_pair = get_eval_pair(sense_index)
    lemma_index = {lemma: idx for idx, lemma in enumerate(lemma_labels)}
    
    data_merged['sense_idx'] = data_merged['formatted_sense_id'].map(sense_index)
    # Problem about it
    # data_merged['lemma_idx'] = data_merged['lemma'].map(lemma_index)
    data_merged['lemma_idx'] = data_merged['formatted_sense_id'].apply(lambda x: lemma_index[x.split('.')[0]])
    data_merged['sense_group'] = data_merged['sense_idx'].map(eval_pair)
    # We keep those columns for now
    keys_to_keep = ['lemma', 'word', 'sentence_text', 'lemma_idx', 'formatted_sense_id', 'sense_idx', 'sense_group']
    data_merged = data_merged[keys_to_keep]

    return data_merged, lemma_pair


def log_cosh_loss(norm_u, norm_v):
    return torch.log(torch.cosh(norm_u - norm_v)).mean()


def load_tree(wsChildrenFile = None):
    """
    Read the file of wsChildren, save them into dictionary
    
    :param wsChildrenFile:path of wsChildrenFile
    :return:two dictionary of word2vecDic and wsChildrenDic
    """
    wsChildrenDic = dict()
    with open(wsChildrenFile, 'r') as chfh:
        for ln in chfh:
            wlst = ln[:-1].split()
            wsChildrenDic[wlst[0]] = wlst[1:]
    return wsChildrenDic
    
    
    




