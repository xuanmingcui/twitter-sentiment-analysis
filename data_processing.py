import demoji
import emoji
import re
import emot

import pandas as pd


def translate_emoticons(text):
    emoticons = emot.emot().emoticons(text)
    if len(emoticons['location']) == 0: return text
    splitted_text = []
    emoticons['location'].insert(0, [0, 0])
    emoticons['location'].append([len(text) - 1, len(text) - 1])
    for i in range(len(emoticons['location']) - 1):
        splitted_text.append(text[emoticons['location'][i][1]:emoticons['location'][i + 1][0]])
        if i < len(emoticons['location']) - 2:
            splitted_text.append(f"[{emoticons['mean'][i]}]")
    return ' '.join(splitted_text)


import emoji
import emot


def translate_emoticons(text, with_brackets):
  emoticons = emot.emot().emoticons(text)
  if len(emoticons['location']) == 0: return text
  splitted_text = []
  emoticons['location'].insert(0,[0,0])
  emoticons['location'].append([len(text)-1,len(text)-1])
  for i in range(len(emoticons['location']) - 1):
    splitted_text.append(text[emoticons['location'][i][1]:emoticons['location'][i+1][0]])
    if i < len(emoticons['location']) - 2:
      if with_brackets:
        splitted_text.append(f"[{emoticons['mean'][i]}]")
      else:
        splitted_text.append(f"{emoticons['mean'][i]}")
  return ' '.join(splitted_text)


def text_cleaning(text: str) -> str:
    # TODO: Make sure the [] doesn't interfere with BERT special tokens
    # sub url with <url>
    result = re.sub(r'\w+:\/\/\S+', '[url]', text)
    # sub email with <email>
    result = re.sub(r'\w+@\w+.[\w+]{2,4}', '[email]', result)
    # translate emoticons first as '<>' might exist in emoticons
    result = translate_emoticons(result, True)
    # translate emoji
    result = emoji.demojize(result, delimiters=("[", "]"))
    # recognize date in format: YYYY-MM-DD or DD/MM/YYYY and sub as <date>
    result = re.sub(
        r'(\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01]))|((0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4})', '[date]',
        result)
    # sub @user123 with <user>
    result = re.sub(r'@\S+', '[user]', result)
    # sub #hashtag123 with <hashtag>
    result = re.sub(r'#\S+', '[hashtag]', result)
    # sub $TSLA with [stock]
    result = re.sub(r'\$\S+', '[stock]', result)
    result = re.sub(r'([\n])+', ' ', result)
    result = result.strip()
    return result


def text_cleaning_notags(text: str) -> str:
    # TODO: Make sure the [] doesn't interfere with BERT special tokens
    # sub url with <url>
    result = re.sub(r'\w+:\/\/\S+', '', text)
    # sub email with <email>
    result = re.sub(r'\w+@\w+.[\w+]{2,4}', '', result)
    # translate emoticons first as '<>' might exist in emoticons
    result = translate_emoticons(result, False)
    # translate emoji
    result = emoji.demojize(result, delimiters=("", ""))
    # recognize date in format: YYYY-MM-DD or DD/MM/YYYY and sub as <date>
    result = re.sub(
        r'(\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01]))|((0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4})', '',
        result)
    # sub @user123 with <user>
    result = re.sub(r'@\S+', '', result)
    # sub #hashtag123 with <hashtag>
    result = re.sub(r'#\S+', '', result)
    # sub $TSLA with [stock]
    result = re.sub(r'\$\S+', '', result)
    result = re.sub(r'([\n])+', ' ', result)
    result = result.strip()
    return result


def tweet_processing(new_path, *args: str):
    df = pd.concat([pd.read_csv(arg, lineterminator='\n') for arg in args])
    df = df.loc[(df['date'] > '2018-01-01') & (df['date'] < '2021-06-30')]
    df = df.drop_duplicates().reset_index(drop=True)
    df['text_processed'] = df['tweet'].apply(text_cleaning)
    df['text_processed_notags'] = df['tweet'].apply(text_cleaning_notags)
    df.sort_values(by=['date'], inplace=True)
    df.to_csv(new_path, index=False)
    return df


if __name__ == '__main__':
    ford = r'/Users/xuanmingcui/Documents/projects/my_twint/my_twint/ford.csv'
    lucid = r'/Users/xuanmingcui/Documents/projects/my_twint/my_twint/lucid.csv'
    tsla_policy = r'/Users/xuanmingcui/Documents/projects/my_twint/my_twint/tsla_policy.csv'
    tsla_product = r'/Users/xuanmingcui/Documents/projects/my_twint/my_twint/tsla_tweets.csv'
    # path2 = r'/Users/xuanmingcui/Documents/projects/my_twint/my_twint/elonmusk_till_20210630.csv'
    list_of_files = [ford, lucid, tsla_policy, tsla_product]
    new_paths = ['data/ford.csv', 'data/lucid.csv', 'data/tsla_policy.csv', 'data/tsla_product.csv']
    for path, new_path in zip(list_of_files, new_paths):
        print(new_path)
        tweet_processing(new_path, path)
