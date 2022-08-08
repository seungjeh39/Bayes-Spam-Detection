# Spam Detection - Seungjeh Lee

import numpy as np
from sklearn.naive_bayes import MultinomialNB



# dictionary to look up words from the data vector (case sensitive)
dictionary = np.array(["you","are","selected","won","lottery","travel",
                       "for","free","credit","cards","very","good","night","send",
                       "us","your","password","account","renew","get","is","congrats"])



# spam data : training set
# sentence vectors
X = np.array([
 [0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],     # [1] travel for free
 [1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],     # [1] congrats you won lottery
 [1,1,1,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],     # [1] you are selected for credit cards
 [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],     # [0] very good
 [0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0],     # [0] good night
 [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],     # [1] lottery
 [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0],     # [1] send us your password
 [0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0],     # [1] get free credit cards
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0],     # [0] renew your account
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0],     # [0] your account is good
 [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]      # [0] congrats you are selected
])

# target values (spam=1, not spam=0)
y = np.array([1,1,1,0,0,1,1,1,0,0,0])



# Set up model and train 
clf = MultinomialNB()
clf.fit(X, y)

# Training result
print("Training Score (accuracy: 1.0 = 100%) = ",end="")
print(clf.score(X,y),"\n")



# testVecToWord: Convert test vectors to words
def testVecToWord(testVec):
    words = ""
    for i in range(len(testVec[0])):
        if testVec[0][i] == 1:
            words += dictionary[i] + " "
    return words

# testSentenceToVec: Convert test sentences to vectors
def testSentenceToVec(sentence):
    testVec = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    words = sentence.split()
    for i in range(len(words)):
        for j in range(len(dictionary)):
            if words[i] == dictionary[j]:
                testVec[0][j] = 1
                break
    return testVec

# Get spam probaility 
def getSpamProbability(test):
    print("Spam test: ", testVecToWord(test))
    print("  answer                     = ", clf.predict(test))
    print("  nospam vs spam probability = ", clf.predict_proba(test))   # [1. 0.] correspond to NoSpam and Spam probability, respectively.
    if clf.predict(test)[0] == 1:
        print("It is a spam")
    elif clf.predict(test)[0] == 0:
        print("It is not a spam")
    print(" ")



# Test sentences
sentence1 = "you won lottery"
test1 = testSentenceToVec(sentence1)

sentence2 = "congrats you won free travel"
test2 = testSentenceToVec(sentence2)

sentence3 = "very good credit cards"
test3 = testSentenceToVec(sentence3)

sentence4 = "good night for you"
test4 = testSentenceToVec(sentence4)

sentence5 = "free travel credit for you"
test5 = testSentenceToVec(sentence5)

sentence6 = "congrats for free credit cards"
test6 = testSentenceToVec(sentence6)

sentence7 = "credit cards are good"
test7 = testSentenceToVec(sentence7)



# Calculate spam probability
getSpamProbability(test1)
getSpamProbability(test2)
getSpamProbability(test3)
getSpamProbability(test4)
getSpamProbability(test5)
getSpamProbability(test6)
getSpamProbability(test7)
