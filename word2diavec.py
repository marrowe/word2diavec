import argparse
import re
# import vector_training
from gensim.models import FastText, Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser()
# Preload saved models, or train some from existing tweet data
parser.add_argument('-fasttext', '--ft', action="store", dest="ft", type=str, default="fasttext.model")
parser.add_argument('-word2vec', '--w2v', action="store", dest="w2v", type=str, default="word2vec.model")
parser.add_argument('-tweet_csv', '--csv', action="store", dest="t", type=str, default="tweet_data.tsv")
parser.add_argument('-load_true', '--load', action="store_true", dest="load", default=True)

parser.add_argument('-text_output', '--in', action="store", dest="outtxt", type=str, default="text_output.txt")
parser.add_argument('-text_input', '--out', action="store", dest="intxt", type=str, default="text_input.txt")
parser.add_argument('-analogies', '--an', action="store", dest="analogies", type=str, default="word-test.v1.txt")
parser.add_argument('-explore_true', '--exp', action="store_false", dest="expl", default=False)
parser = parser.parse_args()


def cosines(model1, model2):
    """
        Takes cosine similarities between identical word pairs in two vector space models,
        then averages the results together and exports them to a .txt file.
        Requires two models trained on the same data (and thus having the same vocab).
    :param model1: KeyedVector object to test
    :param model2: other KeyedVector object to test
    :return: None
    """
    vector_similarity = {}
    vector_average = 0
    for word in model1.vocab:
        vector_similarity[word] = cosine_similarity(model1[word].reshape(1, -1), model2[word].reshape(1, -1))[0][0]
        vector_average += vector_similarity[word]

    vector_average = vector_average / len(model1.vectors)

    print(f'Done with cosine similarity of {len(model1.vocab)} tokens; average {vector_average}')
    return vector_average


def most_similar(model1, model2, toks):
    """
        Calculates the 3 most similar words for a given token in two models, then returns a list of dicts of cosine
        similarity between models for the token + the most similar metric. For qualitative analysis.

    :param model1: KeyedVector object to test
    :param model2: other KeyedVector object to test
    :param toks: list of tokens to compare between models
    :return: dict of tokens with cosine similarities and most similar words
    """

    tok_dict = {}
    for p in toks:
        ft_sim = model1.most_similar(p)
        w2v_sim = model2.most_similar(p)
        tok_dict[p] = (cosine_similarity(model1[p].reshape(1, -1), model2[p].reshape(1, -1))[0][0],
                       ft_sim[0:3], w2v_sim[0:3])

    return tok_dict


def linzen_tests(analogies, model, exploratory=False):
    """
        This function has two modes, regular and exploratory.

    Regular:
        Following Linzen (2016), "Issues in evaluating semantic spaces using word analogies,"
        this function uses "vanilla," "only-b," and "ignore-a" word analogy metrics to measure semantic space accuracy
        by finding the "correct" analogy offset.
            - Vanilla: literal offset method (a:a'::b:__)
            - Only-B: returns the nearest neighbor of b
            - Ignore-A: returns the word most similar to a' and b (a' + b)
        Cosine similarities are calculated for each metric on a four-word analogy pair (e.g., cat:cats::dog:dogs).
        They then are compared with the given b, and given a point if accurate.
        The function returns their total points divided by number of analogies.
    Exploratory:
        Runs the same Linzen metrics as Regular, but instead returns a list of dictionaries of the model's answers.
        Allows for the discovery of what the model "thinks" a word belongs to, such as yall:[placename].
        Activate with exploratory=True.

    :param analogies: list of strings of analogies to test, in the order [a, a', b, __].
    :param model: vector space model to be tested
    :param exploratory: toggles exploratory mode. Defaults to False
    :return: list of accuracy scores for vanilla, only-b, and ignore-a offsets;
             in exploratory mode, list of dicts of tuples of analogy answers
    """

    count = [0, 0, 0]
    explore = []

    for i, tok in enumerate(analogies):
        if not any([t not in model.vocab for t in tok]):  # avoids KeyErrors
            a, ap, b, xp = tok

            if not exploratory:
                if model.most_similar(positive=[ap, b], negative=a)[0][0] == xp:  # vanilla
                    count[0] += 1

                if model.most_similar(b)[0][0] == xp:  # only-b
                    count[1] += 1

                if model.most_similar(positive=[ap, b])[0][0] == xp:  # ignore-a
                    count[2] += 1
            else:
                explore_dict = {f'{a} : {ap} :: {b} : ': (model.most_similar(positive=[ap, b], negative=a)[0][0],
                                                          model.most_similar(b)[0][0],
                                                          model.most_similar(positive=[ap, b])[0][0])}
                explore.append(explore_dict)

        if i % 500 == 0:
            print(f'{i+1} / {len(analogies)} Linzen metrics done')

    if not exploratory:
        try:
            scores = [num / len(analogies) for num in count]
        except ZeroDivisionError:
            print("No analogies found.")
            scores = 0
        return scores
    else:
        return explore


def analogy_parse(analogies, start="", end=""):
    """
        Reads and listifies analogy sets in the form "a:a::b:b'" where the colons are spaces or tabs.
        Allows users to specify where to begin and end in the file in case of colon-delimited sections of interests,
            e.g. at ": city-to-state" in the Miklov et al. (2013) data set.

    :param analogies: path to text file containing line-separated analogy sets.
    :param start: string for section to begin on
    :param end: string to end parsing
    :return: list of lists of strings of tweets
    """
    analogies_list = []
    with open(analogies, 'r') as analogies:
        use = False
        if start == "":
            use = True
        for a in analogies:
            if start == a.strip():
                use = True  # skips to metrics relevant for data
            elif end == a.strip() and end != "":
                break
            if use:
                if ':' in a:
                    continue  # avoid section headers
                else:
                    a = re.sub(' ', '\t', a.strip()).lower()  # normalize data
                    analogies_list.append(a.split('\t'))

    return analogies_list


def main(load_bool):
    """
    :param load_bool: flag to determine whether to open models from provided paths
        or to train new ones using vector_training.
    """
    if not load_bool:
        vector_training.main(parser.t)  # will save to default paths
        ft = FastText.load("fasttext.model").wv
        w2v = Word2Vec.load("word2vec.model").wv
        print("Hello words! Models trained and loaded.")
    else:
        ft = FastText.load(parser.ft).wv
        w2v = Word2Vec.load(parser.w2v).wv
        print("Hello words! Models loaded.")

    analogies_list = analogy_parse(parser.analogies, start="", end="")
    ft_scores = linzen_tests(analogies_list, ft, exploratory=parser.expl)
    w2v_scores = linzen_tests(analogies_list, w2v, exploratory=parser.expl)

    with open(parser.intxt, 'r') as inp:
        pronoun_list = [i.strip() for i in inp]
        top_pronouns = most_similar(ft, w2v, pronoun_list)

    with open(parser.outtxt, 'w') as out:
        out.write(f'Linzen scores\texplore={parser.expl}\n' + '='*25)
        out.write(f'\nFT: {ft_scores}\t\tw2v: {w2v_scores}\n\n\n\n')
        out.write(f'Token similarity\n' + '='*25)
        [out.write(f'\n{p}: {top_pronouns[p][0]}\n'
                   f'\t{top_pronouns[p][1]}'
                   f'\n\t{top_pronouns[p][2]}') for p in top_pronouns]
        print(f'Data saved to {out.name}')


if __name__ == '__main__':
    main(parser.load)
