# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import pandas as pd
from six.moves import urllib
import tensorflow as tf

CSV_COLUMNS = [
    "User_id", "Movie_id", "Rating", "Timestamp", "Action", "Adventure",
    "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama",
    "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-fi", "Thriller", "War", "Western", "Zip-code"
]

user_id = tf.feature_column.categorical_column_with_hash_bucket("User_id", hash_bucket_size=20e4)
zip_code = tf.feature_column.categorical_column_with_hash_bucket("Zip-code", hash_bucket_size=20e4)
action = tf.feature_column.categorical_column_with_vocabulary_list("Action", ["0", "1"])
adventure = tf.feature_column.categorical_column_with_vocabulary_list("Adventure", ["0", "1"])
animation = tf.feature_column.categorical_column_with_vocabulary_list("Animation", ["0", "1"])
children = tf.feature_column.categorical_column_with_vocabulary_list("Children", ["0", "1"])
comedy = tf.feature_column.categorical_column_with_vocabulary_list("comedy", ["0", "1"])
crime = tf.feature_column.categorical_column_with_vocabulary_list("Crime", ["0", "1"])
documentary = tf.feature_column.categorical_column_with_vocabulary_list("Documentary", ["0", "1"])
fantasy = tf.feature_column.categorical_column_with_vocabulary_list("Fantasy", ["0", "1"])
film_noir = tf.feature_column.categorical_column_with_vocabulary_list("Film-Noir", ["0", "1"])
horror = tf.feature_column.categorical_column_with_vocabulary_list("Horror", ["0", "1"])
musical = tf.feature_column.categorical_column_with_vocabulary_list("Musical", ["0", "1"])
mystery = tf.feature_column.categorical_column_with_vocabulary_list("Mystery", ["0", "1"])
romance = tf.feature_column.categorical_column_with_vocabulary_list("romance", ["0", "1"])
sci_fi = tf.feature_column.categorical_column_with_vocabulary_list("Sci-fi", ["0", "1"])
thriller = tf.feature_column.categorical_column_with_vocabulary_list("Thriller", ["0", "1"])
war = tf.feature_column.categorical_column_with_vocabulary_list("War", ["0", "1"])
western = tf.feature_column.categorical_column_with_vocabulary_list("Western", ["0", "1"])

# Wide columns and deep columns.
base_columns = [
    user_id, action, adventure, animation, children, comedy,
    crime, documentary, fantasy, film_noir, horror, musical,
    mystery, romance, sci_fi, thriller, war, western, zip_code
]

genres = tf.feature_column.crossed_column(
    [action, adventure, animation, children, comedy,
     crime, documentary, fantasy, film_noir, horror, musical,
     mystery, romance, sci_fi, thriller, war, western], hash_bucket_size=int(20e4))

userXzip = tf.feature_column.crossed_column(
    [user_id, zip_code], hash_bucket_size=int(20e4))

zipXgeneres = tf.feature_column.crossed_column(
    [zip_code, genres], hash_bucket_size=int(20e4))

crossed_columns = [
    user_id, zip_code, genres, userXzip, zipXgeneres,
    tf.feature_column.crossed_column([user_id, genres], hash_bucket_size=int(20e4)),
    tf.feature_column.crossed_column([user_id, zipXgeneres], hash_bucket_size=int(20e4)),
    tf.feature_column.crossed_column([userXzip, zipXgeneres], hash_bucket_size=int(20e4))
]
deep_columns = [
    tf.feature_column.embedding_column(user_id, dimension=8),
    tf.feature_column.embedding_column(zip_code, dimension=8)
]


# def maybe_download(train_data, test_data):
#  """Maybe downloads training data and returns train and test file names."""
#  if train_data:
#    train_file_name = train_data
#  else:
#    train_file = tempfile.NamedTemporaryFile(delete=False)
#    urllib.request.urlretrieve(
#        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
#        train_file.name)  # pylint: disable=line-too-long
#    train_file_name = train_file.name
#    train_file.close()
#    print("Training data is downloaded to %s" % train_file_name)
#
#  if test_data:
#    test_file_name = test_data
# else:
#   test_file = tempfile.NamedTemporaryFile(delete=False)
#   urllib.request.urlretrieve(
#       "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
#       test_file.name)  # pylint: disable=line-too-long
#   test_file_name = test_file.name
#   test_file.close()
#   print("Test data is downloaded to %s"% test_file_name)

# return train_file_name, test_file_name


def build_estimator(model_dir, model_type):
    """Build an estimator."""
    if model_type == "wide":
        m = tf.estimator.LinearClassifier(
            model_dir=model_dir, feature_columns=base_columns + crossed_columns)
    elif model_type == "deep":
        m = tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=[100, 50])
    else:
        m = tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=crossed_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[100, 50])
    return m


def input_fn(data_file, num_epochs, shuffle):
    """Input builder function."""
    df_data = pd.read_csv(
        tf.gfile.Open(data_file),
        names=CSV_COLUMNS,
        skipinitialspace=True,
        engine="python",
        skiprows=1)
    # remove NaN elements
    df_data = df_data.dropna(how="any", axis=0)
    labels = df_data["income_bracket"].apply(
        lambda x: 1 if (x == 1) else 2 if (x == 2) else 3 if (x == 3) else 4 if (x == 4) else 5).astype(int)
    return tf.estimator.inputs.pandas_input_fn(
        x=df_data,
        y=labels,
        batch_size=100,
        num_epochs=num_epochs,
        shuffle=shuffle,
        num_threads=5)


def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
    """Train and evaluate the model."""
    df_train = pd.read_csv("train.csv", dtype={
        'User_id': str,
        "Action": str,
        "Adventure": str,
        "Animation": str,
        "Children": str,
        "Comedy": str,
        "Crime": str,
        "Drama": str,
        "Fantasy": str,
        "Film-Noir": str,
        "Horror": str,
        "Musical": str,
        "Mystery": str,
        "Romance": str,
        "Sci-fi": str,
        "Thriller": str,
        "War": str,
        "Western": str,
        "Zip-code": str,
        "Documentary": str,
    }, names=CSV_COLUMNS, skipinitialspace=True)
    df_test = pd.read_csv("test.csv", dtype={
        'User_id': str,
        "Action": str,
        "Adventure": str,
        "Animation": str,
        "Children": str,
        "Comedy": str,
        "Crime": str,
        "Drama": str,
        "Fantasy": str,
        "Film-Noir": str,
        "Horror": str,
        "Musical": str,
        "Mystery": str,
        "Romance": str,
        "Sci-fi": str,
        "Thriller": str,
        "War": str,
        "Western": str,
        "Zip-code": str,
        "Documentary": str,
    }, names=CSV_COLUMNS, skipinitialspace=True)
    model_dir = df_train.mkdtemp() if model_dir else model_dir
    #model_type = "wide&deep"
    m = build_estimator(model_dir, model_type)
    m.train(
        input_fn=input_fn(df_train, num_epochs=None, shuffle=True),
        steps=train_steps
    )
    results = m.evaluate(
        input_fn=input_fn(df_test, num_epochs=1, shuffle=False),
        steps=None)
    print("model directory = %s" % model_dir)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))


#FLAGS = None


def main(_):
    #train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
    #               FLAGS.train_data, FLAGS.test_data)
    train_and_eval("/Users/pfang/model/final/model",
                   "wide&deep",
                   "8",
                   "/Users/pfang/model/final/train",
                   "/Users/pfang/model/final/test")

#if __name__ == "__main__":
#    parser = argparse.ArgumentParser()
#    parser.register("type", "bool", lambda v: v.lower() == "true")
#    parser.add_argument(
#        "--model_dir",
#        type=str,
#        default="",
#        help="Base directory for output models."
#    )
#    parser.add_argument(
#        "--model_type",
#        type=str,
#        default="wide_n_deep",
#        help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
#    )
#    parser.add_argument(
#        "--train_steps",
#        type=int,
#        default=2000,
#        help="Number of training steps."
#    )
#    parser.add_argument(
#        "--train_data",
#        type=str,
#        default="",
#        help="Path to the training data."
#    )
#    parser.add_argument(
#        "--test_data",
#        type=str,
#        default="",
#        help="Path to the test data."
#    )
