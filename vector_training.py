# -*- coding: utf-8 -*-
import argparse
import string
import re
from gensim.models import FastText, Word2Vec
from nltk import word_tokenize

parser = argparse.ArgumentParser()
parser.add_argument('-fasttext', '--ft', action="store", dest="ft", type=str, default="fasttext.model")
parser.add_argument('-word2vec', '--w2v', action="store", dest="w2v", type=str, default="word2vec.model")
parser.add_argument('-tweet_tsv', '--tsv', action="store", dest="t", type=str, default="tweet_data.tsv")
parser = parser.parse_args()


def train_models(corpus, ft=True, w2v=True, size=100, window=3, min_count=2, epochs=50):
    """
    From a list of lists of tokenized tweets (or any sentences), trains and saves FastText ("fasttext.model") and
        Word2Vec ("w2v.model") models in current directory.

    :param corpus: list of lists of tokens to train the models on
    :param ft: boolean to train FastText model. Defaults to True
    :param w2v: boolean to train Word2Vec model. Defaults to True
    :param size: number of dimensions for model(s). Defaults to 100
    :param window: context size for model(s). Defaults to 3
    :param min_count: number of times a word has to appear to be trained on. Defaults to 2
    :param epochs: number of epochs for model(s). Defaults to 50
    """
    if ft:
        ftmodel = FastText(size=size, window=window, min_count=min_count)  # instantiate
        ftmodel.build_vocab(sentences=corpus)
        print("Training FastText model...")
        ftmodel.train(sentences=corpus, total_examples=len(corpus), epochs=epochs)
        print(f"Model trained. Time taken: {ftmodel.total_train_time}")
        ftmodel.save(parser.ft)

    if w2v:
        w2vmodel = Word2Vec(corpus, size=size, window=window, min_count=min_count)
        print("Training w2v model...")
        w2vmodel.train(sentences=corpus, total_examples=len(corpus), epochs=epochs)
        print(f"Model trained. Time taken: {w2vmodel.total_train_time}")
        w2vmodel.save(parser.w2v)


def retrain_models(corpus, ft=True, w2v=True):
    """
    Allows preexisting models to continue training for 10 epochs rather than start over entirely.

    :param corpus: list of lists of tokens to train the models on
    :param ft: boolean to train FastText model. Defaults to True
    :param w2v: boolean to train Word2Vec model. Defaults to True
    """
    if ft:
        ftmodel = FastText.load(parser.ft)  # instantiate
        print("Retraining FastText model...")
        ftmodel.train(sentences=corpus, total_examples=len(corpus), epochs=10)
        print(f"Model retrained. Time taken: {ftmodel.total_train_time}")
        ftmodel.save(parser.ft)

    if w2v:
        w2vmodel = Word2Vec.load(parser.w2v)
        print("Retraining w2v model...")
        w2vmodel.train(sentences=corpus, total_examples=len(corpus), epochs=10)
        print(f"Model retrained. Time taken: {w2vmodel.total_train_time}")
        w2vmodel.save(parser.w2v)


def preproc(tweet):
    """
    Preprocesses one tweet for tokenization, with emoji in mind.
    Splits all emoji and special characters into individual tokens to avoid confusing NLTK,
        e.g., into thinking that "yall😭" "yall😂" and "yall😭😭😭😂😂😂" are unique tokens
        rather than variations on the base form "yall".
    Slower than your basic nltk.word_tokenize.

    :param tweet: a single raw string containing tweet text
    :return: list of custom-tokenized tokens in tweet
    """
    punct = string.punctuation + '’¿_*'
    this_tweet = ''
    tweet = re.sub(r'http.*', '', tweet)  # removes urls
    tweet = re.sub(r'\b@[A-Za-z0-9]?\b', '', tweet)  # remove @mentions
    tweet = re.sub(r'\byou (guy|all\b)', r'you\1', tweet)  # for modeling purposes, treat as one token

    emoji_bool = False
    for c in tweet.strip():
        punct_check = any([p for p in punct if p in c])
        if c.isalnum() is False and punct_check is False:
            this_tweet += f' {c} '
            emoji_bool = True
        elif punct_check:  # strips some punctuation to normalize y'all to yall
            if c == "'" or c == "`" or c == "’":
                pass
            else:
                this_tweet += f' {c} '
            emoji_bool = False
        else:
            if emoji_bool:
                this_tweet += f' {c}'
            else:
                this_tweet += c
            emoji_bool = False

    return word_tokenize(this_tweet.lower())


def tokenize(tweet_path, column=6, verbose=True):
    """
    Extracts tweets from a tsv file and tokenizes them using the preproc function.

    :param tweet_path: string of file path to tsv of tweet data
    :param column: column of tsv containing tweet text. Must be the last column of the tsv. Defaults to 6
    :param verbose: prints updates on number of tweets added vs total tweets (minus invalid lines), every 5000 lines.
        Defaults to True
    :return: list of lists of tokenized tweets
    """
    tweet_list = []
    with open(tweet_path) as tw:
        all_content = [t.split('\t') for t in tw]
        loss = 0

        for tweet in all_content[1:]:  # assumes tsv has header
            if len(tweet) == column + 1:  # filters inconsistencies in tsv (e.g., mis-loaded data due to \n in tweet)
                tweet_list.append([t for t in preproc(tweet[column])])
            else:
                loss += 1
            if verbose and len(tweet_list) % 5000 == 0:
                print(f'{len(tweet_list)} of {len(all_content) - loss} tokenized')

    return tweet_list


def main(tweet_tsv, retrain=False):
    """
    Extracts, tokenizes, and trains tweets into FastText and Word2Vec models.

    :param tweet_tsv: tsv of tweets and metadata
    :param retrain: specifies whether data is being
    """
    if not retrain:
        train_models(tokenize(tweet_tsv))
    else:
        retrain_models(tokenize(tweet_tsv))


if __name__ == "__main__":
    main(parser.t)
