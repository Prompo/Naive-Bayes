import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.naive_bayes import BernoulliNB


global x_train_binary


def naive_bayes(m, n, k):
    print("Bernoulli Naive Bayes is running...")
    train_data_results = train_data(m, n, k)
    test_results = naive_bayes_test(m, n, k, train_data_results)


def train_data(m, n, k):
    positive = 0
    negative = 0
    total = 0
    voc = []
    path_neg = "aclImdb/train/neg"
    path_pos = "aclImdb/train/pos"
    vocabulary = create_vocabulary(path_pos, path_neg, m, n, k)
    vectors = create_vectors(path_pos, path_neg, vocabulary)

    # get the probability P(Xi|C)
    print("Its time to calculate P(Xi|C)")
    print("\n")
    P1X0 = get_probabilities(vectors)[0]
    P1X1 = get_probabilities(vectors)[1]

    # get the probability P(C)
    pos = 0
    neg = 0
    total = 0
    for vec in vectors:
        if (vec[0] == 1):
            pos += 1
        else:
            neg += 1
        total += 1

    pos_prob = pos/total
    neg_prob = neg/total

    return P1X0, P1X1, pos_prob, neg_prob, vocabulary


def create_vocabulary(pos_path, neg_path, m, n, k):
    vocabulary = {}
    x_train_binary = []
    counter_list = []
    dictionary = {}
    for file in os.listdir(pos_path):
        with open(os.path.join(pos_path, file), 'r', encoding="utf8") as f:
            text = f.read()
            words = text.split()
            for word in words:
                word = word.strip(","".,/><BR&!").upper()
                if word not in dictionary:
                    dictionary[word] = 1
                else:
                    dictionary[word] += 1

    print("Data from positive reviews were just read")
    print("\n")

    for file in os.listdir(neg_path):
        with open(os.path.join(neg_path, file), 'r', encoding="utf8") as f:
            text = f.read()
            words = text.split()
            for word in words:
                word = word.strip(","".,/><BR&!").upper()
                if word not in dictionary:
                    dictionary[word] = 1
                else:
                    dictionary[word] += 1

    print("Data from negative reviews were just read")
    print("\n")

    for word in dictionary:
        value = dictionary[word]
        if value not in counter_list:
            counter_list.append(value)

    counter_list.sort()

    most_frequent = n
    most_rare = k
    demanded = m

    frequent = {}
    rare = {}

    i = 0
    while len(rare) < most_rare:
        for word in dictionary:
            if dictionary[word] == counter_list[i]:
                rare[word] = 1
                if len(rare) == most_rare:
                    break
        i += 1

    print("Rarest words were found")
    print("\n")

    counter_list.sort(reverse=True)
    i = 0

    while len(frequent) < most_frequent:
        for word in dictionary:
            if dictionary[word] == counter_list[i]:
                frequent[word] = 1
            if len(frequent) == most_frequent:
                break
        i += 1

    print("Most frequent words were found")
    print("\n")

    counter_list.sort(reverse=True)

    freq = True
    ok = True

    for count in counter_list:
        for word in dictionary:
            if (dictionary[word] == count) and ok:
                if (word not in frequent):
                    vocabulary[word] = 1
                    if len(vocabulary) == demanded:
                        ok = False
                        break
        if (ok == False):
            break

    """
    print("rare words:")
    print(len(rare))
    print("frequent words")
    print(len(frequent))
    print("vocabulary:")
    print(len(vocabulary))
    print("\n")
    """

    print("vocabulary is ready!!")

    return vocabulary


def create_vectors(pos_path, neg_path, vocabulary):
    x_train_binary = []
    vectors = []
    for file in os.listdir(pos_path):
        with open(os.path.join(pos_path, file), 'r', encoding="utf8") as f:
            temp = []
            temp.append(1)
            text = f.read()
            words = text.split()
            for word in words:
                word = word.strip(","".,/><BR&!").upper()
                if word in vocabulary:
                    temp.append(1)
                else:
                    temp.append(0)
        vectors.append(temp)
        x_train_binary.append(temp)

    for file in os.listdir(neg_path):
        with open(os.path.join(neg_path, file), 'r', encoding="utf8") as f:
            temp = []
            temp.append(0)
            text = f.read()
            words = text.split()
            for word in words:
                word = word.strip(","".,/><BR&!").upper()
                if word in vocabulary:
                    temp.append(1)
                else:
                    temp.append(0)
        vectors.append(temp)
        x_train_binary.append(temp)

    print("Every text is coded in vectors...")
    print("\n")

    return vectors


def get_probabilities(vectors):
    positive_review = []
    negative_review = []
    PT = 0  # positive text
    NT = 0  # negative text

    max0 = -1
    max1 = -1

    for vec in vectors:

        if (vec[0] == 0):
            NT += 1  # negative texts
        else:
            PT += 1  # positive texts
        if (len(vec) > max0) and vec[0] == 0:
            max0 = len(vec)
        if len(vec) > max1 and vec[0] == 1:
            max1 = len(vec)

    if (max0 > max1):
        max = max0
    else:
        max = max1

    P1C0 = {}
    P1C1 = {}
    for i in range(1, max):
        P1C0[i] = 0

    for i in range(1, max):
        P1C1[i] = 0

    for vec in vectors:
        i = 1
        while i < len(vec):
            if vec[0] == 0 and vec[i] == 1:
                P1C0[i] += 1
            elif vec[0] == 1 and vec[i] == 1:
                P1C1[i] += 1
            i += 1

    sum_positive = 0
    sum_negative = 0
    # laplace
    for i in range(1, len(P1C0)):
        neg_prob = (P1C0[i]+1)/(NT+2)  # Negative probability for the Feature
        negative_review.append(neg_prob)
    for i in range(1, len(P1C1)):
        pos_prob = (P1C1[i]+1)/(PT+2)  # Positive probability for the  Feature
        positive_review.append(pos_prob)

    return positive_review, negative_review


def naive_bayes_test(m, n, k, train_vectors):
    print("P(Xi|C) is calculated and now we create vectors for the test data")
    print("\n")
    path_neg = "aclImdb/test/neg"
    path_pos = "aclImdb/test/pos"
    voc = train_vectors[4]
    test_vectors = create_vectors(path_pos, path_neg, voc)
    P1C1 = train_vectors[0]
    P1C0 = train_vectors[1]

    TP = 0  # true_positive
    TN = 0  # true_negative
    FP = 0  # false_positive
    FN = 0  # false_negative

    position = []
    ACCURACY = []
    PRECISION = []
    RECALL = []
    F1 = []

    counter = 1
    signal = False
    for test in test_vectors:
        priori_pos = 1
        priori_neg = 1
        for i in range(1, len(test)):
            if (i < len(P1C1)):
                if (test[i] == 1):
                    priori_pos *= P1C1[i]
                    priori_pos *= P1C0[i]
                else:
                    priori_neg *= (1-P1C1[i])
                    priori_neg *= (1-P1C0[i])

        # priori_pos*=train_vectors[2]
        # priori_neg*=train_vectors[3]

        if (priori_pos > priori_neg) == test[0]:
            if priori_pos > priori_neg:
                TP += 1
            else:
                TN += 1
        else:
            if priori_pos > priori_neg:
                FP += 1
            else:
                FN += 1
        if (TP > 0) and (TN > 0) and (FP > 0) and (FN > 0):
            signal = True
        if (signal):
            PRECISION.append((TP/(TP+FP)))
            ACCURACY.append((TP+TN)/(TP+TN+FN+FP))
            RECALL.append((TP/(TP+FN)))
            F1.append(2*(TP/(TP+FP))*(TP/(TP+FN))/((TP/(TP+FP))+(TP/(TP+FN))))
            position.append(counter)

        counter += 1

    print(TP)
    print(TN)
    print(FP)
    print(FN)
    print("\n")
    print("position")
    print(len(position))

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    accuracy = (TP+TN)/(TP+TN+FN+FP)
    f1 = 2*precision*recall/(precision+recall)
    print("\n")
    print("Accuracy")
    print(accuracy)
    print("\n")
    print("Precision:")
    print(precision)
    print("\n")
    print("Recall:")
    print(recall)
    print("\n")
    print("F1:")
    print(f1)

    fig = plt.figure()
    fig.suptitle("Naive Bayes")

    plt.subplot(2, 2, 1)
    plt.plot(position, ACCURACY)
    plt.xlabel("training data")
    plt.ylabel("Accuracy (%)")

    plt.subplot(2, 2, 2)
    plt.plot(position, PRECISION)
    plt.xlabel('training data')
    plt.ylabel("Precision (%)")

    plt.subplot(2, 2, 3)
    plt.plot(position, RECALL)
    plt.xlabel("training data")
    plt.ylabel("Recall (%)")

    plt.subplot(2, 2, 4)
    plt.plot(position, F1)
    plt.xlabel("training data")
    plt.ylabel("F1 (%)")

    plt.show()

    return TP, TN, FP, FN


naive_bayes(10000, 100, 100)
