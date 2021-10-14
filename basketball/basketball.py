from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy
import pandas


# feed in the data, I adjusted the data format so that we had a clear target column
scores_df = pandas.read_excel('bball_scores.xlsx')

X = scores_df[['Team', 'Opponent']]
y = scores_df['Outcome']

# we need to map the team names to numerical values before feeding into the model
team_mappings = {'Houston Rockets': 0, 'Dallas Mavericks': 1, 'Boston Celtics': 2, 'Toronto Raptors': 3, 'Denver Nuggets': 4}
X['Team'] = X['Team'].map(team_mappings)
X['Opponent'] = X['Opponent'].map(team_mappings)

# check to make sure mapping is correct
# print(X)

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.10, random_state=42)

lr_model = LogisticRegression(penalty='l2', solver='lbfgs').fit(X_train, y_train)
print(lr_model.classes_)
''' 
now going to fit to a logistic regression model, this returns the probability that one class or the other is true, 
thought this was the best model since we are dealing with a binary classification and want the output to be a probability

using l2 penalty, l1 penalty can make sparse models but I don't think we need this here as there are not many features. I
do not think we have a lot of coefficients to deal with 

used the lbfgs solver as an optimization parameter because this is a fairly small data set, lbfgs only keeps a few vectors in memory at a time
This algo starts with initial estimate of the minimum and tries to refine the minimum over iterations
uses derivative of its function to find direction of steepest descent and to predict the matrix that helps it adjust course
'''

# would like to try and evaluate the model
# classes are as follows [ -1 (loss) 1 (win)]
y_scores = lr_model.predict_proba(x_test)


for result, score in zip(y_test, y_scores):
    print(score[0], 'chance of loss')
    print(score[1], 'chance of win')
    print(result, max(score))


'''
Compared actual result of game to class with max probability from my model's prediction
Only the last two were predicted correctly, in a second iteration I would probably generate more data samples
or see if there are any additional features we can add to make more accurate predictions. 

0.355236100604567 chance of loss
0.644763899395433 chance of win
-1 0.644763899395433

0.41689111165213333 chance of loss
0.5831088883478667 chance of win
-1 0.5831088883478667

0.6915131431138914 chance of loss
0.3084868568861085 chance of win
1 0.6915131431138914

0.48963086021019986 chance of loss
0.5103691397898001 chance of win *
1 0.5103691397898001

0.5113401520242109 chance of loss
0.48865984797578904 chance of win *
-1 0.5113401520242109
'''

# now predicting the test case
test_case = numpy.array([4, 1]).reshape(1, -1)
nugsvsmavs = lr_model.predict_proba(test_case)
print(nugsvsmavs)
'''
The model shows a 76% chance of the result belonging to class -1, that is there's a 76% chance the Nuggets will lose.
[[0.76035131 0.23964869]]
'''
