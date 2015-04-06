from sklearn import tree
from collections import Counter
from string import punctuation, whitespace
import csv


def feature_extract(title):
    features = []

    # Title length
    features.append(len(title))

    # Does it start with uppercase?
    features.append(int(title[0:1].isupper()))

    # Do all words start with uppercase?
    features.append(int(title.istitle()))

    # How many punctuation characters?
    features.append(
        sum([v for k, v in Counter(title).items() if k in punctuation])
    )

    # How many whitespace characters?
    features.append(
        sum([v for k, v in Counter(title).items() if k in whitespace])
    )

    # How many whitespace separated words?
    features.append(len(title.split()))

    # Does it split into title / subtitle?
    title_subtitle = list(filter(lambda e: e, title.split(': ')))

    features.append(
        len(title_subtitle)
    )

    # Assuming it splits into title/subtitle, is title uppercase?
    features.append(
        len(title_subtitle) > 0 and title_subtitle[0].istitle()
    )

    # Assuming it splits into title/subtitle, is subtitle uppercase?
    features.append(
        len(title_subtitle) > 1 and title_subtitle[1].istitle()
    )

    return features

data = [
    ('', 0)
]

tests = []

with open('samples.csv', 'r') as csv_data:
    reader = csv.reader(csv_data, delimiter="\t")
    for row in reader:
        if row[1] in ['0', '1']:
            data.append((row[2], int(row[1])))
        else:
            tests.append(row[2])

features = list(map(lambda entry: feature_extract(entry[0]), data))
classes = list(map(lambda entry: entry[1], data))

#print("Class -> feature list:")
#for klass, feature in zip(classes, features):
#    print(klass, " -> ", feature)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(list(features), classes)

for t in tests:
    print(t, ' -> ', clf.predict(feature_extract(t)))
