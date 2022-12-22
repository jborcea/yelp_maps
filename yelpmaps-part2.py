"""A Yelp-powered Restaurant Recommendation Program"""

from unicodedata import name
from abstractions import *
from data import ALL_RESTAURANTS, CATEGORIES, USER_FILES, load_user_file
from ucb import main, trace, interact
from utils import distance, mean, zip, enumerate, sample
from visualize import draw_map

##################################
# Phase 2: Unsupervised Learning #
##################################


def find_closest(location, centroids):
    """Return the centroid in centroids that is closest to location. If
    multiple centroids are equally close, return the first one.

    >>> find_closest([3.0, 4.0], [[0.0, 0.0], [2.0, 3.0], [4.0, 3.0], [5.0, 5.0]])
    [2.0, 3.0]
    """
    # BEGIN Question 3
    
    return min(centroids, key=lambda x: distance(x,location))

    # END Question 3


def group_by_first(pairs):
    """Return a list of pairs that relates each unique key in the [key, value]
    pairs to a list of all values that appear paired with that key.

    Arguments:
    pairs -- a sequence of pairs

    >>> example = [ [1, 2], [3, 2], [2, 4], [1, 3], [3, 1], [1, 2] ]
    >>> group_by_first(example)
    [[2, 3, 2], [2, 1], [4]]
    """
    keys = []
    for key, _ in pairs:
        if key not in keys:
            keys.append(key)
    return [[y for x, y in pairs if x == key] for key in keys]


def group_by_centroid(restaurants, centroids):
    """Return a list of clusters, where each cluster contains all restaurants
    nearest to a corresponding centroid in centroids. Each item in
    restaurants should appear once in the result, along with the other
    restaurants closest to the same centroid.
    >>> r1 = make_restaurant('X', [4, 3], [], 3, [
    ...         make_review('X', 4.5),
    ...      ]) # r1's location is [4,3]
    >>> r2 = make_restaurant('Y', [-2, -4], [], 4, [
    ...         make_review('Y', 3),
    ...         make_review('Y', 5),
    ...      ]) # r2's location is [-2, -4]
    >>> r3 = make_restaurant('Z', [-1, 2], [], 2, [
    ...         make_review('Z', 4)
    ...      ]) # r3's location is [-1, 2]
    >>> c1 = [4, 5]
    >>> c2 = [0, 0]
    >>> groups = group_by_centroid([r1, r2, r3], [c1, c2])
    >>> [[restaurant_name(r) for r in g] for g in groups]
    [['X'], ['Y', 'Z']] # r1 is closest to c1, r2 and r3 are closer to c2
    """
    # BEGIN Question 4

    rest_location = [restaurant_location(restaurant) for restaurant in restaurants ]
    closest = [find_closest(i, centroids) for i in rest_location]
    zipped = zip(closest, restaurants)
    return group_by_first(zipped)

    # END Question 4


def find_centroid(cluster):
    """Return the centroid of the locations of the restaurants in cluster.
    >>> r1 = make_restaurant('X', [4, 3], [], 3, [
    ...         make_review('X', 4.5),
    ...      ]) # r1's location is [4,3]
    >>> r2 = make_restaurant('Y', [-3, 1], [], 4, [
    ...         make_review('Y', 3),
    ...         make_review('Y', 5),
    ...      ]) # r2's location is [-3, 1]
    >>> r3 = make_restaurant('Z', [-1, 2], [], 2, [
    ...         make_review('Z', 4)
    ...      ]) # r3's location is [-1, 2]
    >>> cluster = [r1, r2, r3]
    >>> find_centroid(cluster)
    [0.0, 2.0]
    """
    # BEGIN Question 5
    
    rest_location = [restaurant_location(restaurant) for restaurant in cluster]
    x = []
    y = []
    for i in rest_location:
        get_x = i[0]
        get_y = i[1]
        x.append(get_x)
        y.append(get_y)
    return [mean(x), mean(y)]
        

    # END Question 5


def k_means(restaurants, k, max_updates=100):
    """Use k-means to group restaurants by location into k clusters."""
    assert len(restaurants) >= k, 'Not enough restaurants to cluster'
    previous_centroids = [] 
    n = 0
    # Select initial centroids randomly by choosing k different restaurants

    # Complete the implementation of k_means. In each iteration of the while statement,
    # 1. Group restaurants into clusters, where each cluster contains all restaurants closest to the same centroid. (Hint: Use group_by_centroid)
    # 2. Bind centroids to a new list of the centroids of all the clusters. (Hint: Use find_centroid)
   
    centroids = [restaurant_location(r) for r in sample(restaurants, k)]

    while previous_centroids != centroids and n < max_updates:
        previous_centroids = centroids
        # BEGIN Question 6
        clusters = group_by_centroid(restaurants, centroids)
        centroids = [find_centroid(cluster) for cluster in clusters]
        # END Question 6
        n += 1
    return centroids 


def find_predictor(user, restaurants, feature_fn):
    """Return a score predictor (a function from restaurants to scores),
    for a user by performing least-squares linear regression using feature_fn
    on the items in restaurants. Also, return the R^2 value of this model.

    Arguments:
    user -- A user
    restaurants -- A sequence of restaurants
    feature_fn -- A function that takes a restaurant and returns a number
    """
    reviews_by_user = {review_restaurant_name(review): review_score(review)
                       for review in user_reviews(user).values()}

    xs = [feature_fn(r) for r in restaurants] #the extracted values for each restaurant in restaurants
    ys = [reviews_by_user[restaurant_name(r)] for r in restaurants] #the names for the restaurants in restaurants

    # BEGIN Question 7
    s_xx = 0
    s_yy = 0
    s_xy = 0
    for i in range(len(xs)):
        s_xx += (xs[i] - mean(xs))**2
        s_yy += (ys[i] - mean(ys))**2
        s_xy += (xs[i] - mean(xs))*(ys[i] - mean(ys))

    b = s_xy/s_xx
    a = mean(ys)-b*mean(xs)
    r_squared = (s_xy)**2/(s_xx*s_yy)
    
    # END Question 7

    def predictor(restaurant):
        return b * feature_fn(restaurant) + a

    return predictor, r_squared


def best_predictor(user, restaurants, feature_fns):
    """Find the feature within feature_fns that gives the highest R^2 value
    for predicting scores by the user; return a predictor using that feature.



    Arguments:
    user -- A user
    restaurants -- A list of restaurants
    feature_fns -- A sequence of functions that each takes a restaurant
    """
    reviewed = user_reviewed_restaurants(user, restaurants)

    # BEGIN Question 8
    # Implement best_predictor, which takes a user, a list of restaurants, and a sequence of feature_fns. 
    # It uses each feature function to compute a predictor function, then returns the predictor that has the highest r_squared value.
    # All predictors are learned from the subset of restaurants reviewed by the user (called reviewed in the starter implementation).
    
    predictor_list = []
    rsquared_list = []
    for i in feature_fns:
        predict,rsquared = find_predictor(user, reviewed, i)
        predictor_list.append(predict)
        rsquared_list.append(rsquared)
    zipped_list = zip(predictor_list, rsquared_list)
    return max(zipped_list, key=lambda predict: predict[1])[0]


    # END Question 8


def rate_all(user, restaurants, feature_fns):
    """Return the predicted scores of restaurants by user using the best
    predictor based a function from feature_fns.

    Arguments:
    user -- A user
    restaurants -- A list of restaurants
    feature_fns -- A sequence of feature functions
    """
    predictor = best_predictor(user, ALL_RESTAURANTS, feature_fns)
    reviewed = user_reviewed_restaurants(user, restaurants)

    # BEGIN Question 9
    
    predicted = {}
    for restaurant in restaurants:
        if restaurant in reviewed:
            predicted[restaurant_name(restaurant)] = user_score(user, restaurant_name(restaurant))
        else:
            predicted[restaurant_name(restaurant)] = predictor(restaurant)
    return predicted

    # END Question 9


def search(query, restaurants):
    """Return each restaurant in restaurants that has query as a category.

    Arguments:
    query -- A string
    restaurants -- A sequence of restaurants
    """
    # BEGIN Question 10
    
    return [restaurant for restaurant in restaurants if query in restaurant_categories(restaurant)]

    # END Question 10


def feature_set():
    """Return a sequence of feature functions."""
    return [restaurant_mean_score,
            restaurant_price,
            restaurant_num_scores,
            lambda r: restaurant_location(r)[0],
            lambda r: restaurant_location(r)[1]]


@main
def main(*args):
    import argparse
    parser = argparse.ArgumentParser(
        description='Run Recommendations',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-u', '--user', type=str, choices=USER_FILES,
                        default='test_user',
                        metavar='USER',
                        help='user file, e.g.\n' +
                        '{{{}}}'.format(','.join(sample(USER_FILES, 3))))
    parser.add_argument('-k', '--k', type=int, help='for k-means')
    parser.add_argument('-q', '--query', choices=CATEGORIES,
                        metavar='QUERY',
                        help='search for restaurants by category e.g.\n'
                        '{{{}}}'.format(','.join(sample(CATEGORIES, 3))))
    parser.add_argument('-p', '--predict', action='store_true',
                        help='predict scores for all restaurants')
    parser.add_argument('-r', '--restaurants', action='store_true',
                        help='outputs a list of restaurant names')
    args = parser.parse_args()

    # Output a list of restaurant names
    if args.restaurants:
        print('Restaurant names:')
        for restaurant in sorted(ALL_RESTAURANTS, key=restaurant_name):
            print(repr(restaurant_name(restaurant)))
        exit(0)

    # Select restaurants using a category query
    if args.query:
        restaurants = search(args.query, ALL_RESTAURANTS)
    else:
        restaurants = ALL_RESTAURANTS

    # Load a user
    assert args.user, 'A --user is required to draw a map'
    user = load_user_file('{}.dat'.format(args.user))

    # Collect ratings
    if args.predict:
        ratings = rate_all(user, restaurants, feature_set())
    else:
        restaurants = user_reviewed_restaurants(user, restaurants)
        names = [restaurant_name(r) for r in restaurants]
        ratings = {name: user_score(user, name) for name in names}

    # Draw the visualization
    if args.k:
        centroids = k_means(restaurants, min(args.k, len(restaurants)))
    else:
        centroids = [restaurant_location(r) for r in restaurants]
    draw_map(centroids, restaurants, ratings)
