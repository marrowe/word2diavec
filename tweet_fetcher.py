# -*- coding: utf-8 -*-

""" ------------------------------------------------------------------------------
tweet_fetcher.py

    a simple implementation of Tweepy for Python 3
    that downloads tweets & user info from Twitter search queries into tsv format.

Originally written for LING-447, American Dialects, as tweet_sentiment_to_csv.py

    by Margaret Anne Rowe, with ample input from Ethan Beaman
        mar299                  |                  ejb100
                        @ georgetown.edu

Rewritten and revamped for LING-472, Python for Computational Linguistics

    entirely by M.A. Rowe

Last updated: April 29, 2019

------------------------------------------------------------------------------"""

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler, TweepError, API, Cursor

import csv
import re
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('-key_document', '--keys', action="store", dest="k", type=str, default="api_key.txt")
parser.add_argument('-output_file', '--output', action="store", dest="o", type=str, default="tweet_data.tsv")
parser.add_argument('-token_list', '--toks', action="store", dest="t", type=str, default="tok_list.txt")
parser.add_argument('-geographic_locations', '--geo', action="store", dest="g", type=str, default="geotargets.txt")
parser.add_argument('-number_to_download', '--download', action="store", dest="num", type=int, default=1000)
parser = parser.parse_args()

metadata_fields = ['created_at', 'lang', 'screen_name', 'description', 'location']


class MyStreamListener(StreamListener):
    # Initializes the Tweepy StreamListener object used to fetch tweets from Twitter.

    def on_data(self, status):
        return True

    def on_error(self, status_code):
        if status_code == 420:
            print("ERROR: failed to connect")
            return False


class TweetFetch:
    def __init__(self, key_dict, query_list, fetch_num=1000, make_csv=True):
        self.tweets = self.scraper(self.start_api(key_dict), query_list, fetch_num)
        if make_csv:
            self.write_to_csv(self.tweets, parser.o)

    def start_api(self, key_dict):  # must be provided by user - freely available @ developer.twitter.com
        consumer_key = key_dict['cons']
        consumer_secret_key = key_dict['cons_sec']
        access_token = key_dict['acc']
        access_token_secret = key_dict['acc_sec']

        auth = OAuthHandler(consumer_key, consumer_secret_key)
        auth.set_access_token(access_token, access_token_secret)

        try:
            auth.get_authorization_url()
            print("API connected.")
        except TweepError:
            print("ERROR: API did not connect")
            quit()

        return API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    def scraper(self, api, queries, fetch_num):
        """--------------------------------------------------------------------------
            Collects tweets for analysis using a Tweepy cursor object and user-defined search terms.

        :param api: initialized Twitter API object
        :param queries: list of string searches to be run on Twitter
        :param fetch_num: number of tweets to return per search term
        :return: set of tweet tuples, each containing tuple metadata specified in get_data
        --------------------------------------------------------------------------"""
        # my_queries: List of search terms to look for. Follows standard Twitter syntax
        # -filter:retweets excludes all RTs - recommended for sentiment analysis

        all_tweets = set()

        for search in queries:
            search += " -filter:retweets"
            query = Cursor(api.search, lang='en', rpp=100, tweet_mode='extended', q=search)
            print(f"Gathering tweets for '{search}'...")
            results = query.items(fetch_num)
            temp_list = []
            try:  # Using specifications in get_data(), saves desired metadata fields of an individual tweet.
                temp_list = list(map(self.get_data, [status._json for status in results]))
                time.sleep(5)
            except TweepError:
                print(f"Error detected on {search}")
                pass
            for t in temp_list:  # Converts tweet metadata into tuples, then adds to set of all downloaded tweets
                all_tweets.add(tuple(t))

        return self.filter(all_tweets)

    def get_data(self, status, geo=False):
        """--------------------------------------------------------------------------
            Extracts datetime, @handle, display name, location, description, and tweet text
            from the user dict of individual tweet metadata.

        :param status: dict of metadata of one raw tweet
        :param geo: boolean to fetch/not fetch geographic coordinates
        :return: tuple of specified tweet metadata and text of tweet
        --------------------------------------------------------------------------"""
        user = status.get('user')  # user is a dictionary within status - contains metadata of tweeter
        data = {f: user.get(f) for f in metadata_fields}  # extracts metadata specified in constant
        if geo:
            if status.get('geo') is not None:  # gets geodata - rarely used
                geodude = status.get('geo')['coordinates']  # NB: contributes to rate limit!
                data.update({'geo': f'UT: {geodude[0]},{geodude[1]})'})
            else:
                data.update({'geo': None})
        data.update({'text': status.get('full_text')})  # extracts text of actual tweet
        return data.items()

    def filter(self, tweet_tup):
        """--------------------------------------------------------------------------
            Filters tweets for user-defined words/phrases in individual tweet tuples. Meant to pare down
            results using criteria beyond the original search terms, e.g., user locations.
            Adds sentiment analysis of tweet text using TextBlob's sentiment() function.

        :param tweet_tup: set of tweet tuples that have already been filtered for desired metadata fields.
        :return: list of dictionaries of tweets that successfully matched the criteria
        --------------------------------------------------------------------------"""
        us_locations = set()
        with open(parser.g) as places:  # locations to match
            for line in places:
                us_locations.add(line.strip())
            us_locations.remove('')  # in case of null

        # stoplist of non-location characters, words, or confounds (e.g., CHI->ATL)
        filter_chara = ['✈️', '/', '✈', '-', '➡', '\|', '\bto\b', '\bhell\b', '⚣', '\bs?he/', '\bthey', '->']

        saved_tweets = []

        col = len(metadata_fields) - 1

        for tweet in tweet_tup:
            # if '\n' in [n[1] for n in tweet]:              # this should fix the linebreak issue - unfortunately
            #     [re.sub('\n', '\r', n[1]) for n in tweet]  # couldn't get it to work in time for the deadline
            if tweet[col][1] is not "" and tweet[col][1] is not None:  # make sure tweet location isn't null
                if any(re.search(r'\b' + place + r'\b', tweet[col][1], re.I) for place in list(us_locations)):
                    if not any(re.search(f, tweet[col][1], re.I) for f in filter_chara):
                        current_tweet = dict(tweet)
                        saved_tweets.append(current_tweet)
        return saved_tweets

    def write_to_csv(self, final_tweets, path):
        """
            Saves gathered tweet data in tsv format.

        :param final_tweets: dictionary of tweets to be saved in tsv format
        :param path: desired output filename
        :return: None
        """
        with open(path, 'w', encoding='utf-8') as tsvfile:
            fieldnames = metadata_fields + ['geo'] + ['text']
            writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=fieldnames, lineterminator='\n')
            writer.writeheader()
            writer.writerows(final_tweets)
            print(f"All done! Your tweets have been saved to {path}")

        return None


def main(num=parser.num):
    api_dict = {}
    with open(parser.k) as keys:
        for tokens in keys:
            tokens = tokens.split('\t')
            api_dict[tokens[0]] = tokens[1].strip()

    with open(parser.t) as toks:
        tok_list = [tok for tok in toks]

    TweetFetch(api_dict, tok_list, num)


if __name__ == "__main__":
    main(parser.num)
