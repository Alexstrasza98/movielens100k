import pandas as pd
from datetime import datetime
import time
import uszipcode
import re
from collections import defaultdict
import torch
from tqdm import tqdm

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def title2bert(movie_path):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)

    movies = pd.read_csv(movie_path, sep="\|", encoding='ISO-8859-1', engine='python', names=[])
    movies = movies.iloc[:, [0, 1]]
    movies.columns = ["movie_id", "title"]
    for _, (movie_id, title) in tqdm(movies.iterrows()):
        title = title[:-6]
        for target_piece in [", The", ", A"]:
            if re.search(target_piece, title):
                title = title.replace(target_piece, " ")
                title = target_piece[2:] + " " + title

        tokens = tokenizer(title, return_tensors="pt")

        with torch.no_grad():
            hidden = model(**tokens)
            cls = hidden.last_hidden_state[:, 0, :]

        torch.save(cls, f"./data/BERT_features/{movie_id}.pt")


def zipcode2area(sr, zipcode, mode="county"):
    # assume all zip codes are from US
    address = sr.by_zipcode(zipcode)
    # some outliers (53 out of 943)
    if address is None:
        return None

    if mode == "county":
        return address.county
    elif mode == "state":
        return address.state
    else:
        return None


def create_dataframe(rating_path, movie_path, user_path):
    """
    Read and merge data from rating, movie and user to construct input data frame
    :param rating_path: path to rating data file
    :param movie_path: path to movie data file
    :param user_path: path to user data file
    :return: input features for predicting the ratings
    """
    # Read ratings data
    columns_name = ["user_id", "item_id", "rating", "timestamp"]
    ratings_df = pd.read_csv(rating_path, sep="\t", names=columns_name)

    # Read movie data
    columns_name = ["movie_id", "movie_title", "release_date", "video_release_date",
                    "IMDB_URL", "unknown", "Action", "Adventure", "Animation",
                    "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                    "FilmNoir", "Horror", "Musical", "Mystery", "Romance", "SciFi",
                    "Thriller", "War", "Western"]
    movie_df = pd.read_csv(movie_path, sep="\|", encoding='ISO-8859-1', engine='python', names=columns_name)
    # drop video_release_date since they are all None, unknown since 99% of this column are 0, IMDB_URL apparently
    movie_df.drop(["video_release_date", "unknown", "IMDB_URL"], axis=1, inplace=True)

    # Read user data
    columns_name = ["user_id", "age", "gender", "occupation", "zip_code"]
    user_df = pd.read_csv(user_path, sep="\|", engine="python", names=columns_name)

    # Filter missing samples in movies and users
    missing_movie = movie_df[movie_df.isnull().any(axis=1)].movie_id
    movie_df.drop(movie_df[movie_df.movie_id.isin(missing_movie)].index, axis=0, inplace=True)
    ratings_df.drop(ratings_df[ratings_df.item_id.isin(missing_movie)].index, axis=0, inplace=True)

    missing_user = user_df[user_df.isnull().any(axis=1)].user_id
    user_df.drop(user_df[user_df.user_id.isin(missing_user)].index, axis=0, inplace=True)
    ratings_df.drop(ratings_df[ratings_df.user_id.isin(missing_user)].index, axis=0, inplace=True)

    # Add extra features (average_rating and total_reviews) for movie and user
    extra_features = ratings_df[["item_id", "rating"]].groupby("item_id").agg(movie_average_rating=("rating", "mean"),
                                                                              movie_total_reviews=("rating", "count"))
    movie_df = pd.merge(movie_df, extra_features, how="left", left_on="movie_id", right_on="item_id")

    extra_features = ratings_df[["user_id", "rating"]].groupby("user_id").agg(user_average_rating=("rating", "mean"),
                                                                              user_total_reviews=("rating", "count"))
    user_df = pd.merge(user_df, extra_features, how="left", left_on="user_id", right_on="user_id")

    # Change format of movie_title (remove year) and date (into Unix timestamp)
    movie_df["movie_title"] = movie_df["movie_title"].apply(lambda x: x[:-6])
    movie_df["release_date"] = movie_df["release_date"].apply(
        lambda x: int(time.mktime(datetime.strptime(x, "%d-%b-%Y").timetuple())))

    # Convert zipcode to county or state (county is still too sparse according to analysis)
    sr = uszipcode.SearchEngine()
    user_df["zip_code"] = user_df["zip_code"].apply(lambda x: zipcode2area(sr, x, mode="state"))

    # Merge all information
    dataset = pd.merge(ratings_df, movie_df, how="left", left_on="item_id", right_on="movie_id")
    dataset = pd.merge(dataset, user_df, how="left", left_on="user_id", right_on="user_id")

    # Drop ids (leave user id for computing NDCG score)
    dataset.drop(["item_id", "movie_id"], axis=1, inplace=True)

    return dataset


def feature_tags(feature_names):
    # current features can be divided into two parts:
    # numeric ones
    # categorical ones:
    #   - gender, occupation, zip_code, movie genre
    tags = ["numeric", "gender", "occupation", "zip_code", "genre"]

    tags2cols = defaultdict(list)
    for tag in tags:
        for i, feature in enumerate(feature_names):
            if re.search(tag, feature):
                tags2cols[tag].append(i)

    # make sure all columns are selected
    assert sum(map(len, tags2cols.values())) == len(feature_names)

    return tags2cols


def prepare_data(rating_path_train, rating_path_test, movie_path, user_path, validation=None, random=None):
    train_df = create_dataframe(rating_path_train, movie_path, user_path)
    train_labels = train_df["rating"]
    train_ids = train_df["user_id"]
    train_df.drop(["user_id", "rating"], axis=1, inplace=True)

    test_df = create_dataframe(rating_path_test, movie_path, user_path)
    test_labels = test_df["rating"]
    test_ids = test_df["user_id"]
    test_df.drop(["user_id", "rating"], axis=1, inplace=True)

    num_vars = ["timestamp", "release_date", "movie_average_rating", "movie_total_reviews", "age",
                "user_average_rating", "user_total_reviews"]
    cat_vars = ["gender", "occupation", "zip_code"]
    remain_vars = ["Action", "Adventure", "Animation",
                   "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                   "FilmNoir", "Horror", "Musical", "Mystery", "Romance", "SciFi",
                   "Thriller", "War", "Western",
                   # "movie_title",
                   ]

    numeric_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("one-hot", OneHotEncoder(sparse=False))
    ])

    remain_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])

    data_pipeline = ColumnTransformer(transformers=[
        ("numerical", numeric_pipeline, num_vars),
        ("categorical", categorical_pipeline, cat_vars),
        ("genre", remain_pipeline, remain_vars)
    ])

    train_input = data_pipeline.fit_transform(train_df)
    test_input = data_pipeline.transform(test_df)
    val_input, val_labels, val_ids = None, None, None
    names = data_pipeline.get_feature_names_out()
    if validation is not None:
        train_input, val_input, train_labels, val_labels, train_ids, val_ids = train_test_split(train_input,
                                                                                                train_labels,
                                                                                                train_ids,
                                                                                                shuffle=True,
                                                                                                test_size=validation,
                                                                                                random_state=random)

    return {"train": (train_input, train_labels, train_ids),
            "val": (val_input, val_labels, val_ids),
            "test": (test_input, test_labels, test_ids),
            "feature_tags": feature_tags(names)}
