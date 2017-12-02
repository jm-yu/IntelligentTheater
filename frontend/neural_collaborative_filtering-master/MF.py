import argparse
from time import time

import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds


def parse_args():
    parser = argparse.ArgumentParser(description="Run MF.")
    parser.add_argument('--mode', nargs='?', default='predict',
                        help='eval or predict?')
    return parser.parse_args()

def recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations=5):
    # Get and sort the user's predictions
    user_row_number = userID - 1  # UserID starts at 1, not 0
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False)  # UserID starts at 1

    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.UserID == (userID)]
    user_full = (user_data.merge(movies_df, how='left', left_on='MovieID', right_on='MovieID').
                 sort_values(['Rating'], ascending=False)
                 )

    print 'User {0} has already rated {1} movies.'.format(userID, user_full.shape[0])
    print 'Recommending highest {0} predicted ratings movies not already rated.'.format(num_recommendations)

    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies_df[~movies_df['MovieID'].isin(user_full['MovieID'])].
                       merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left',
                             left_on='MovieID',
                             right_on='MovieID').
                       rename(columns={user_row_number: 'Predictions'}).
                       sort_values('Predictions', ascending=False).
                       iloc[:num_recommendations, :-1]
                       )

    return user_full, recommendations

def eval(predictions_df, userID, movies_df, test_df, original_ratings_df, num_recommendations=5):
    user_row_number = userID  # UserID starts at 1, not 0
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False)  # UserID starts at 1

    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.UserID == (userID)]
    user_full = (user_data.merge(movies_df, how='left', left_on='MovieID', right_on='MovieID').
                 sort_values(['Rating'], ascending=False)
                 )
    test_data = test_df[test_df.UserID == (userID)]
    test = test_data.merge(movies_df, how='left', left_on='MovieID', right_on='MovieID')
    test_list = movies_df[movies_df['MovieID'].isin(test['MovieID'])]
    check_list = movies_df[~movies_df['MovieID'].isin(user_full['MovieID'])].sample(n=99)
    #print check_list
    #print test_list
    check_list = check_list.append(test_list)
    while len(check_list)!=100:
        check_list = movies_df[~movies_df['MovieID'].isin(user_full['MovieID'])].sample(n=99)
        check_list = check_list.append(test_list)

    #print len(check_list)


    #Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (check_list.
                       merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left',
                             left_on='MovieID',
                             right_on='MovieID').
                       rename(columns={user_row_number: 'Predictions'}).
                       sort_values('Predictions', ascending=False).
                       iloc[:num_recommendations, :-1]
                       )
    #print recommendations
    pre = recommendations['MovieID'].values.tolist()
    test = test_list['MovieID'].values.tolist()
    #print pre
    #print test
    for i in range(len(pre)):
        #print(1)
        if pre[i]==test[0]:
            return 1
    return 0

def create_eval_data():
    rating_list = []
    test_list = []
    with open('Data/ml-1m/ratings.dat', "r") as f:
        line = f.readline()
        cur = 1
        while line != None and line != "":
            arr = line.split("::")
            user, movie, rating, time = int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3])
            if user!=cur:
                test_list.append(rating_list.pop())
                cur = user
            rating_list.append([user, movie, rating, time])
            line = f.readline()
        test_list.append(rating_list.pop())
    #print(test_list)
    movies_list = [i.strip().split("::") for i in open('Data/ml-1m/movies.dat', 'r').readlines()]
    return rating_list, movies_list, test_list

def create_predict_data():
    ratings_list = [i.strip().split("::") for i in open('/Applications/MAMP/htdocs/movie_recommendation/resources/test.csv', 'r').readlines()]
    movies_list = [i.strip().split("::") for i in open('/Applications/MAMP/htdocs/movie_recommendation/resources/test.csv', 'r').readlines()]
    with open('Data/input.csv', "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split(",")
            item, rating = int(arr[0]), int(arr[1])
            user = int(6041)
            if item>65132:
                item = item%65132
            time = 1111
            ratings_list.append([user, item, rating, time])
            line = f.readline()
    return ratings_list, movies_list

if __name__ == '__main__':
    args = parse_args()
    mode = args.mode
    if mode == 'predict':
        print 'load data'
        create_predict_data()
        print 'finish load data'
        ratings_list, movies_list = create_predict_data()
        ratings = np.array(ratings_list)
        movies = np.array(movies_list)
        ratings_df = pd.DataFrame(ratings_list, columns=['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype=int)
        movies_df = pd.DataFrame(movies_list, columns=['MovieID', 'Title', 'Genres'])
        movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)
        R_df = ratings_df.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)
        R = R_df.as_matrix()
        user_ratings_mean = np.mean(R, axis=1)
        R_demeaned = R - user_ratings_mean.reshape(-1, 1)
        U, sigma, Vt = svds(R_demeaned, k=50)
        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        preds_df = pd.DataFrame(all_user_predicted_ratings, columns=R_df.columns)
        print 'get predictions:'
        already_rated, predictions = recommend_movies(preds_df, 6041, movies_df, ratings_df, 10)
        # eval(preds_df, 1, movies_df, test_df, ratings_df, 10)
        with open('/Applications/MAMP/htdocs/movie_recommendation/resources/rank.txt', "w") as f:
            f.write(str(predictions['MovieID'].values.tolist()))
        print predictions
    if mode == 'eval':
        ratings_list, movies_list, test_list = create_eval_data()
        ratings = np.array(ratings_list)
        movies = np.array(movies_list)
        test = np.array(test_list)
        ratings_df = pd.DataFrame(ratings_list, columns=['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype=int)
        test_df = pd.DataFrame(test_list, columns=['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype=int)
        movies_df = pd.DataFrame(movies_list, columns=['MovieID', 'Title', 'Genres'])
        movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)
        R_df = ratings_df.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)
        R = R_df.as_matrix()
        user_ratings_mean = np.mean(R, axis=1)
        R_demeaned = R - user_ratings_mean.reshape(-1, 1)
        U, sigma, Vt = svds(R_demeaned, k=50)
        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        preds_df = pd.DataFrame(all_user_predicted_ratings, columns=R_df.columns)
        predictions = 0
        for i in range(3706):
            # print i+1
            t1 = time()
            predictions = predictions + eval(preds_df, i + 1, movies_df, test_df, ratings_df, 10)
            print "loop :{0}, time {1}".format(i, time() - t1)
        print '{0:.3f}'.format(float(predictions) / 3706)
