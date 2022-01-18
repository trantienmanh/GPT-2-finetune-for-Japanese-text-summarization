import re
from typing import Dict, List
import pandas as pd
import unicodedata
import json, os
from utils.utils import logger
import argparse
from sklearn.model_selection import train_test_split

def read_jsonl_data(file_path: str):
    with open(file=file_path, mode='r', encoding='utf-8') as f:
        data = [json.loads(r.strip()) for r in f.readlines()]

        return data

def clean_text(text: str) -> str:

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ぁ-んァ-ン一-龥ーA-Za-z0-9「」。、・\uFF08\uFF09(),!?%&$+='`―-]", "", text)

    return text.lower().strip()

def clean_target(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub("\n{2,}", "\n", text)
    text = re.sub("\n", "。", text)
    text = re.sub(r"[ぁ-んァ-ン一-龥ーA-Za-z0-9「」。、・\uFF08\uFF09(),!?%&$+='`―-]", "", text)
    return text.lower().strip()

def create_data_frame(data: List[Dict]):
    df = pd.DataFrame(data=data)

    df['raw_source'] = df.source
    df['raw_target'] = df.target
    logger.info('Cleaning text...')
    df.source = df.source.apply(lambda x: clean_text(x))
    df.target = df.target.apply(lambda x: clean_target(x))

    logger.info("Splitting train, val, and test data...")
    X_train, X_test = train_test_split(df, test_size=10000, random_state=42, shuffle=True)
    X_train, X_val = train_test_split(X_train, test_size=10000, random_state=42, shuffle=False)
    df_split = pd.concat([X_train, X_val, X_test], axis=0)

    is_train = [True]*len(X_train) + [False]*(len(X_val)+len(X_test))
    is_val = [False]*len(X_train) + [True]*(len(X_val)) + [False]*len(X_test)
    is_test = [False]*(len(X_train)+len(X_val)) + [True]*len(X_test)

    df_split['is_train'] = is_train
    df_split['is_val'] = is_val
    df_split['is_test'] = is_test
    logger.info("Validating data...")
    df_split = df_split[(df_split.source!='')&(df_split.target!='')]
    
    return df_split

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--raw_data_path', type=str, default='./data/summarization_data_extend_.jsonl', help='raw data path')
    parser.add_argument('--storage_dir', type=str, default="./data/", help="Storage path")
    parser.add_argument("--file_name", type=str, default="jp_text_sum_extend.csv")
    args = parser.parse_args()

    logger.info(f"Reading raw data: {args.raw_data_path}")
    data = read_jsonl_data(args.raw_data_path)

    df = create_data_frame(data=data)

    logger.info(f"Saving processed data to {args.storage_dir}")
    df.to_csv(os.path.join(args.storage_dir, args.file_name))