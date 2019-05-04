# word2diavec
***Final project for LING-472 Python for Computational Linguistics***

This project examines vector space similarity for dialectal second person plural pronouns in American English.
It trains `gensim` FastText and Word2Vec models on tweets, then calculates cosine similarity between models and tokens.

Data was gathered from [Twitter](https://www.twitter.com) using an implementation of [Tweepy](http://www.tweepy.org/),
originally written for LING-452 American Dialects as [tweet_sentiment_to_csv.py](https://github.com/marrowe/tweet-sentiment)
but rewritten for this project.

### Required libraries:
* [gensim](https://radimrehurek.com/gensim/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [NLTK](http://www.nltk.org/)
* [Tweepy](http://www.tweepy.org/) (tweet_fetcher.py only)

Runs in Python 3.6+.


## word2diavec.py

word2diavec evaluates the performance of two vector space models trained on the same data
by calculating the average cosine similarities of their vectors
using analogy metrics proposed by Tal Linzen (2016). 
It can take two pretrained models or a dataset to be trained in vector_training.py.

Normally, the model first saves average cosine similarity between weights in models' shared vocabulary. 
Then, using a list of analogies in the form a:a* :: b:b*, 
it calculates cosine similarities between "most similar" word offsets to provide the missing word, 
e.g., talk:talking::swim:swimming. 
If the model correctly provides b*, it receives a point, but it receives no points for those it gets wrong.

Metrics used:
1. "vanilla" offset method
2. only-b: nearest neighbor cosine similarities of b
3. ignore-a: most similar to both a* and b

It also returns a the cosine similarities between models for a list of target words, 
as well as the top three most similar words for each model.

In Explore mode, all of the above is true, except instead of returning the scored Linzen metrics,
it returns a dictionary of the analogy in question with the model's Linzen results in a tuple
(e.g., {'a : a* :: b : ': (vanilla word, only-b word, ignore-a word)

Output consists of the average Linzen scores and individual cosine similarities in .txt format.

Example usage with loaded models, not in Explore mode:
`word2diavec.py --ft fasttext.model --w2v word2vec.model --in text_input.txt --out text_output.txt --an word-test.v1.txt
--l`

In Explore mode:
`word2diavec.py --e --in text_input.txt --out text_output.txt`

Output consists of a .txt file with the results of the four metrics.

## vector_training.py

This script uses `gensim` implementations of FastText and word2vec to train models on a corpus of tweets.
It uses a custom tokenization function that converts concatenations (e.g., 'you guys') to one word ('youguys') 
and that treats emojis as individual words, e.g., so as to avoid thinking that 
"yall😭" "yall😂" and "yall😭😭😭😂😂😂" are unique tokens rather than variants of the word "y'all." 

It can also retrain preexisting models for 10 additional epochs.

Example usage:
`vector_training.py --ft fasttext.model --w2v word2vec.model --tsv tweet_data.tsv`

Output consists of trained FastText and word2vec .model files.

## tweet_fetcher.py

This script uses an implementation of Joshua Roesslein's [Tweepy](http://www.tweepy.org/) (v.3.7.0)
to download tweets & user metadata from Twitter search queries, exclude those not tweeted in certain areas, 
and save the data in .tsv format.

Originally written for LING-447 as [tweet_sentiment_to_csv.py](https://github.com/marrowe/tweet-sentiment),
it has been almost entirely rewritten to improve performance and usefulness.

To use this Twitter scraper, you will need free API keys available from their [Developer website](https://developer.twitter.com/).
Once connected, the scraper will retrieve a user-specified number of tweets per user-specified search term.
It then filters tweets based on a user-provided list of locations, 
which can be supplemented by user-provided stopwords/tokens,
keeping only tweets whose location fields contain the desired locations with no stopwords.

Output consists of tweet metadata and tweets in .tsv format.

Example usage:
`tweet_fetcher.py --keys api_key.txt --output tweet_data.tsv --geo geotargets.txt --toks tok_list.txt --num 1000`

## Results
Overall performance of the models on the test data were quite poor, 
likely due in part to the small sample size after filtering. 
tweet_fetcher.py did not properly save tweets or tweet metadata containing line breaks, 
causing a large swath of data to be unusable.

The FastText model fared best on 2/3 Linzen (2016) metrics on Mikolov et al. (2013)'s data, vs. Word2Vec. 
ALL is the average of all metrics, while SYN is the average of the syntactic categories only.

    Average cosine similarity b/w models: -0.0400
    
    Linzen offset metrics
                    FT      w2v
    VANILLA (ALL) 	0.0174	0.0012
    VANILLA (SYN)	0.0315	0.0017
    ONLY-B (ALL)	0.0491	0.0055
    ONLY-B (SYN)	0.0852	0.0058
    IGNORE-A (ALL)	0.1055	0.0138
    IGNORE-A (SYN)	0.1891	0.0215

When tested on pronoun_analogies.py, scores were also quite low:

    Linzen offset metrics
                FT          w2v
    VANILLA     0.0100	0.0300
    ONLY-B	    0.0250	0.0000
    IGNORE-A    0.0100	0.0000

And cosine similarities between models on the same pronouns showed them to be fairly unrelated: 

    Pronoun	    cos sim.
    you         -0.0604
    youse	    -0.0246
    yinz	    0.0100
    yall	    0.0236
    youguys     -0.0315
    youall	    0.0731

Cosine similarity between pronouns was slightly better:

                            FastText
    pronoun	    youse   yinz    yall    youguys youall
    you         0.6966  0.6781  0.7798  0.7917  0.7266
    youse	    ------  0.6954  0.5937  0.5641  0.5214
    yinz        ------  ------  0.6704  0.6168  0.5422
    yall        ------  ------  ------  0.7290  0.5993
    youguys     ------  ------  ------  ------  0.7523
    
                            word2vec
    pronoun	    youse   yinz    yall    youguys youall
    you         0.5651  0.6444  0.7628  0.7167  0.6401
    youse       ------  0.7007  0.4959  0.4523  0.3748
    yinz        ------  ------  0.6420  0.5716  0.5222
    yall        ------  ------  ------  0.5969  0.5192
    youguys	    ------  ------  ------  ------  0.5969


See report for a breakdown of scores by category 
and a further explanation of processes and shortcomings.


## Future goals
I hope to add data visualization tools to this repository in the near future, 
such as implementations of mapping programs or vector space visualizers, 
to make the results more accessible to non-specialists 
and more thoroughly explore the "dialect" component of the project.