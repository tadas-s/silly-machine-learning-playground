from sklearn import tree
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from collections import Counter, OrderedDict
from string import punctuation, whitespace
import numpy as np
import csv
import re
from functools import reduce


def feature_extract(title):
    features = []

    # Title length
    features.append(len(title))

    # Does it start with uppercase?
    features.append(int(title[0:1].isupper()))

    # How many punctuation characters?
    features.append(
        sum([v for k, v in Counter(title).items() if k in punctuation])
    )

    # How many less common punctuation characters?
    features.append(
        sum([v for k, v in Counter(title).items() if (k in punctuation) and (k not in '.,:\'')])
    )

    # Does it end with punctuation?
    features.append(
        int(len(title) and title[-1] in punctuation)
    )

    # How many whitespace characters?
    features.append(
        sum([v for k, v in Counter(title).items() if k in whitespace])
    )

    # How many whitespace separated words?
    features.append(len(title.split()))

    # What's an average length of uppercase words?
    upper_lengths = list(map(len, filter(lambda t: t[:1].isalpha() and t[:1].isupper(), title.split())))
    if len(upper_lengths):
        features.append(np.average(upper_lengths))
    else:
        features.append(0.0)

    # What's an average length of lowercase words?
    lower_lengths = list(map(len, filter(lambda t: t[:1].isalpha() and t[:1].islower(), title.split())))
    if len(lower_lengths):
        features.append(np.average(lower_lengths))
    else:
        features.append(0.0)

    # Does it split into title / subtitle?
    title_subtitle = list(filter(lambda e: e, title.split(': ')))

    features.append(
        len(title_subtitle)
    )

    # Assuming it splits into title/subtitle, is subtitle starts with uppercase?
    features.append(
        len(title_subtitle) > 1 and title_subtitle[1][0:1].isupper()
    )

    # Is title/subtitel split by " : "?
    features.append(
        int(" : " in title)
    )

    # Does it have any kind of text in parenthesis?
    features.append(
        len(re.findall("\[[^\]]*\]|\([^)]*\)|<[^>]*>|\{[^\}]*\}", title))
    )

    # Uppercase letter 'share'
    uppercase = reduce(lambda l, e: l + 1, filter(lambda l: l.isalpha() and l.isupper(), title), 0)
    if uppercase > 0:
        features.append(float(len(title)) / float(uppercase))
    else:
        features.append(0.0)

    # Lowercase letter 'share'
    lowercase = reduce(lambda l, e: l+1, filter(lambda l: l.isalpha() and l.islower(), title), 0)
    if lowercase > 0:
        features.append(float(len(title)) / float(lowercase))
    else:
        features.append(0.0)

    return features


def load_demo_data(filename):
    data = {}

    with open(filename) as csv_data:
        reader = csv.reader(csv_data, delimiter=",")

        for row in reader:
            isbn = row[1]
            data.setdefault(isbn, [])
            data[isbn].append(row[2])

    return OrderedDict(sorted(data.items()))

data = [
    ('', 0)
]

tests = []

with open('sample_multivendor_responses.csv', 'r') as csv_data:
    reader = csv.reader(csv_data, delimiter=",")
    for row in reader:
        if row[2] in ['0', '1', '2']:
            data.append((row[3], int(row[2])))
        else:
            tests.append(row[3])

features = list(map(lambda entry: feature_extract(entry[0]), data))
classes = list(map(lambda entry: entry[1], data))

#print("Class -> feature list:")
#for klass, feature in zip(classes, features):
#    print(klass, " -> ", feature)

#clf = tree.DecisionTreeClassifier()
#clf = BernoulliNB()
clf = RandomForestClassifier(min_samples_split=4)


clf = clf.fit(list(features), classes)

tests = load_demo_data('test_multivendor_responses.csv')

for isbn, candidates in tests.items():
    probs = list(map(lambda c: (c, clf.predict_proba(feature_extract(c))[0]), candidates))
    probs = sorted(probs, key=lambda v: (-max(v[1][1], v[1][2]), v[1][0]))
    print("Candidates:")
    print("\n".join(sorted(candidates)))
    print("\nWinner:")
    print(probs[0][0])
    print("------------------------------------------------")