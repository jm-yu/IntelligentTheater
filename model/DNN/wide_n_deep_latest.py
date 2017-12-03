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
"""Example code for TensorFlow Wide & Deep Tutorial using TF High Level API.
This example uses APIs in Tensorflow 1.4 or above.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys
import tempfile

import pandas as pd
from six.moves import urllib
import tensorflow as tf


CSV_COLUMNS = [
    'user', 'movie', 'rating', 'time', 'action', 'adventure', 'animation', 'children', 'comedy',
    'crime', 'docu', 'drama', 'fantasy', 'filmnoir', 'horror', 'musical', 'mystery', 'romance',
    'scifi', 'thriller', 'war', 'western', 'zip'
]

User = tf.feature_column.numeric_column("user")
Movie = tf.feature_column.numeric_column("movie")
Timestamp = tf.feature_column.numeric_column("time")
Action = tf.feature_column.categorical_column_with_hash_bucket("action", hash_bucket_size=2)
Adventure = tf.feature_column.categorical_column_with_hash_bucket("adventure", hash_bucket_size=2)
Animation = tf.feature_column.categorical_column_with_hash_bucket("animation", hash_bucket_size=2)
Children = tf.feature_column.categorical_column_with_hash_bucket("children", hash_bucket_size=2)
Comedy = tf.feature_column.categorical_column_with_hash_bucket("comedy", hash_bucket_size=2)
Crime = tf.feature_column.categorical_column_with_hash_bucket("crime", hash_bucket_size=2)
Documentary = tf.feature_column.categorical_column_with_hash_bucket("docu", hash_bucket_size=2)
Drama = tf.feature_column.categorical_column_with_hash_bucket("drama", hash_bucket_size=2)
Fantasy = tf.feature_column.categorical_column_with_hash_bucket("fantasy", hash_bucket_size=2)
Film_Noir = tf.feature_column.categorical_column_with_hash_bucket("filmnoir", hash_bucket_size=2)
Horror = tf.feature_column.categorical_column_with_hash_bucket("horror", hash_bucket_size=2)
Musical = tf.feature_column.categorical_column_with_hash_bucket("musical", hash_bucket_size=2)
Mystery = tf.feature_column.categorical_column_with_hash_bucket("mystery", hash_bucket_size=2)
Romance = tf.feature_column.categorical_column_with_hash_bucket("romance", hash_bucket_size=2)
Sci_fi = tf.feature_column.categorical_column_with_hash_bucket("scifi", hash_bucket_size=2)
Thriller = tf.feature_column.categorical_column_with_hash_bucket("thriller", hash_bucket_size=2)
War = tf.feature_column.categorical_column_with_hash_bucket("war", hash_bucket_size=2)
Western = tf.feature_column.categorical_column_with_hash_bucket("western", hash_bucket_size=2)
Zip_code = tf.feature_column.numeric_column("zip")

# Wide columns and deep columns.
base_columns = [
      User, Movie, Action, Adventure, Animation, Children, Comedy, Crime, Documentary, Drama,
      Fantasy, Film_Noir, Horror, Musical, Mystery, Romance, Sci_fi, Thriller, War, Western
]

crossed_columns = [
      tf.feature_column.crossed_column(['action', 'adventure', 'animation', 'children', 'comedy', 'crime',
                                        'docu', 'drama', 'fantasy', 'filmnoir', 'horror', 'musical', 'mystery',
                                        'romance', 'scifi', 'thriller', 'war', 'western'], hash_bucket_size=1e4)
]

deep_columns = [
      User,
      Movie,
      # To show an example of embedding
      tf.feature_column.embedding_column(Action, dimension=8),
      tf.feature_column.embedding_column(Adventure, dimension=8),
      tf.feature_column.embedding_column(Animation, dimension=8),
      tf.feature_column.embedding_column(Children, dimension=8),
      tf.feature_column.embedding_column(Comedy, dimension=8),
      tf.feature_column.embedding_column(Crime, dimension=8),
      tf.feature_column.embedding_column(Documentary, dimension=8),
      tf.feature_column.embedding_column(Drama, dimension=8),
      tf.feature_column.embedding_column(Fantasy, dimension=8),
      tf.feature_column.embedding_column(Film_Noir, dimension=8),
      tf.feature_column.embedding_column(Horror, dimension=8),
      tf.feature_column.embedding_column(Musical, dimension=8),
      tf.feature_column.embedding_column(Mystery, dimension=8),
      tf.feature_column.embedding_column(Romance, dimension=8),
      tf.feature_column.embedding_column(Sci_fi, dimension=8),
      tf.feature_column.embedding_column(Thriller, dimension=8),
      tf.feature_column.embedding_column(War, dimension=8),
      tf.feature_column.embedding_column(Western, dimension=8),
]


FLAGS = None

def build_estimator(model_dir, model_type):
  """Build an estimator."""
  if model_type == "wide":
    m = tf.estimator.LinearClassifier(
        n_classes=5,
        label_vocabulary=['1', '2', '3', '4', '5'],
        model_dir=model_dir,
        feature_columns=base_columns + crossed_columns
    )
  elif model_type == "deep":
    m = tf.estimator.DNNClassifier(
        n_classes=5,
        label_vocabulary=['1', '2', '3', '4', '5'],
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=[100, 75, 50, 25])
  else:
    m = tf.estimator.DNNLinearCombinedClassifier(
        n_classes=5,
        label_vocabulary=['1','2','3','4','5'],
        model_dir=model_dir,
        linear_feature_columns=base_columns + crossed_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 75, 50, 25]
        )
        #tf.estimator.DNNLinearCombinedClassifier()
  return m


def input_fn(data_file, num_epochs, shuffle):
  """Returns an `input_fn` required by Estimator train/evaluate.
  Args:
    data_file: The file path to the dataset.
    num_epochs: Number of epochs to iterate over data. If `None`, `input_fn`
      will generate infinite stream of data.
    shuffle: bool, whether to read the data in random order.
  """
  df_data = pd.read_csv(
      tf.gfile.Open(data_file),
      dtype={
             'user': int,
             'movie': int,
             'rating': str,
             'time': int,
             "action": str,
             "adventure": str,
             "animation": str,
             "children": str,
             "comedy": str,
             "crime": str,
             "drama": str,
             "docu": str,
             "fantasy": str,
             "filmnoir": str,
             "horror": str,
             "musical": str,
             "mystery": str,
             "romance": str,
             "scifi": str,
             "thriller": str,
             "war": str,
             "western": str,
             "zip": str,
         },
      names=CSV_COLUMNS,
      skipinitialspace=True,
      engine="python",
      skiprows=0)
  # remove NaN elements
  df_data = df_data.dropna(how="any", axis=0)
  labels = df_data["rating"].apply(lambda x: 1 if(x=='1') else 2 if(x=='2') else 3 if(x=='3') else 4 if(x=='4') else 5).astype(str)
  #labels = df_data["rating"].apply(lambda x: float(x)/5.0).astype(float)
  #print(labels)
  return tf.estimator.inputs.pandas_input_fn(
      x=df_data,
      y=labels,
      batch_size=100,
      num_epochs=num_epochs,
      shuffle=shuffle,
      num_threads=1)


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  mode = FLAGS.mode
  if mode=='train':
      train_file_name, test_file_name = FLAGS.train_data, FLAGS.test_data
      model_dir = FLAGS.model_dir if FLAGS.model_dir else tempfile.mkdtemp()
      estimator = build_estimator(model_dir, FLAGS.model_type)
      train_spec = tf.estimator.TrainSpec(input_fn=input_fn(train_file_name, num_epochs=None, shuffle=True),
                                          max_steps=FLAGS.train_steps)

      eval_spec = tf.estimator.EvalSpec(input_fn=input_fn(test_file_name, num_epochs=1, shuffle=False), steps=None)
      # print(estimator.type, train_spec.type, eval_spec.type)
      tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
      predictions = estimator.predict(input_fn("testfile.csv", num_epochs=1, shuffle=False), predict_keys='classes')
      # Manual cleanup
      result = list(predictions)
      #predicted_classes = [p["classes"] for p in result]

      # for i, p in enumerate(predictions):
      #    print("Prediction %s: %s" % (i + 1, p))
      # rating = predicted_classes[:,0]
      #print(predicted_classes)
      shutil.rmtree(model_dir)

  if mode=='eval':
      dist = {0:0, 1:0, 2:0, 3:0, 4:0}
      model_dir = FLAGS.model_dir
      estimator = build_estimator(model_dir, FLAGS.model_type)
      predictions = estimator.predict(input_fn("testfile.csv", num_epochs=1, shuffle=False), predict_keys='classes')
      rating = []
      with open("testfile.csv", "r") as f:
          line = f.readline()
          while line != None and line != "":
              arr = line.split(",")
              rating.append(arr[2])
              line = f.readline()
      result = list(predictions)
      predicted_classes = [p["classes"] for p in result]
      miss_sum = 0
      for i in range(len(predicted_classes)):
          miss = abs(int(predicted_classes[i][0])-int(rating[i]))
          miss_sum = miss_sum + miss
          dist[miss] = dist[miss]+1
      print("error distribution: {0}".format(dist))
      print("average error: {0}".format(float(miss_sum)/float(len(predicted_classes))))
      # for i, p in enumerate(predictions):
      #    print("Prediction %s: %s" % (i + 1, p))
      # rating = predicted_classes[:,0]
      #print(predicted_classes)

  if mode=='predict':
      model_dir = FLAGS.model_dir
      estimator = build_estimator(model_dir, FLAGS.model_type)
      #predictions = estimator.predict(input_fn("testfile.csv", num_epochs=1, shuffle=False), predict_keys='classes')

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--mode",
      type=str,
      default="train",
      help="Base directory for output models."
  )
  parser.add_argument(
      "--model_dir",
      type=str,
      default="model/",
      help="Base directory for output models."
  )
  parser.add_argument(
      "--model_type",
      type=str,
      default="wide_n_deep",
      help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
  )
  parser.add_argument(
      "--train_steps",
      type=int,
      default=2000,
      help="Number of training steps."
  )
  parser.add_argument(
      "--train_data",
      type=str,
      default="trainfile.csv",
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data",
      type=str,
      default="testfile.csv",
      help="Path to the test data."
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)