'''
Created on Aug 9, 2016
Keras Implementation of Neural Matrix Factorization (NeuMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import numpy as np
import scipy.sparse as sp
import heapq
from random import shuffle
import theano
import theano.tensor as T
import keras
from keras.models import load_model
from keras import backend as K
from keras import initializations
from keras.regularizers import l1, l2, l1l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import sys
import GMF, MLP
import Unseen
import argparse

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--mode', nargs='?', default='predict',
                        help='train or predict?')
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='10m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for MF embeddings.')                    
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--mf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    return parser.parse_args()

def getUnseen(list1, list2):
    temp1 = []
    temp1 = list1
    temp2 = []
    temp2 = list2
    for i in range(len(temp2)):
        if temp1.__contains__(temp2[i]):
            print(temp2[i])
            temp1.remove(temp2[i])

    return temp1

def load_rating_file_as_matrix(filename):
    '''
    Read .rating file and Return dok matrix.
    The first line of .rating file is: num_users\t num_items
    '''
    # Get number of users and items
    num_users, num_items = 6001, 65132
    # with open(filename, "r") as f:
    #     line = f.readline()
    #     while line != None and line != "":
    #         arr = line.split("\t")
    #         u, i = int(arr[0]), int(arr[1])
    #         num_users = max(num_users, u)
    #         num_items = max(num_items, i)
    #         line = f.readline()
    # Construct matrix
    mat = sp.dok_matrix((num_users + 2, num_items + 1), dtype=np.float32)
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split(",")
            user, item, rating = 6001, int(arr[0]), float(arr[1])
            if item>65132:
                item = item%65132
            if (rating > 0):
                mat[user, item] = 1.0
            line = f.readline()
    return mat


def load_rating_file_as_list(filename):
    ratingList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split(",")
            item = int(arr[0])
            if item>65132:
                item = item%65132
            ratingList.append(item)
            line = f.readline()
    return ratingList

def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)

def get_model(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    
    # Embedding layer
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user',
                                  init = init_normal, W_regularizer = l2(reg_mf), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'mf_embedding_item',
                                  init = init_normal, W_regularizer = l2(reg_mf), input_length=1)   

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = layers[0]/2, name = "mlp_embedding_user",
                                  init = init_normal, W_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = layers[0]/2, name = 'mlp_embedding_item',
                                  init = init_normal, W_regularizer = l2(reg_layers[0]), input_length=1)   
    
    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    mf_vector = merge([mf_user_latent, mf_item_latent], mode = 'mul') # element-wise multiply

    # MLP part 
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    mlp_vector = merge([mlp_user_latent, mlp_item_latent], mode = 'concat')
    for idx in xrange(1, num_layer):
        layer = Dense(layers[idx], W_regularizer= l2(reg_layers[idx]), activation='relu', name="layer%d" %idx)
        mlp_vector = layer(mlp_vector)

    # Concatenate MF and MLP parts
    #mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
    #mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
    predict_vector = merge([mf_vector, mlp_vector], mode = 'concat')
    
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = "prediction")(predict_vector)
    
    model = Model(input=[user_input, item_input], 
                  output=prediction)
    
    return model

def load_pretrain_model(model, gmf_model, mlp_model, num_layers):
    # MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
    model.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
    model.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)
    
    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)
    
    # MLP layers
    for i in xrange(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer('layer%d' %i).get_weights()
        model.get_layer('layer%d' %i).set_weights(mlp_layer_weights)
        
    # Prediction weights
    gmf_prediction = gmf_model.get_layer('prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('prediction').set_weights([0.5*new_weights, 0.5*new_b])    
    return model

def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in xrange(num_negatives):
            j = np.random.randint(num_items)
            while train.has_key((u, j)):
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

if __name__ == '__main__':
    args = parse_args()
    mode = args.mode
    num_epochs = args.epochs
    batch_size = args.batch_size
    mf_dim = args.num_factors
    layers = eval(args.layers)
    reg_mf = args.reg_mf
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learning_rate = args.lr
    learner = args.learner
    verbose = args.verbose
    mf_pretrain = args.mf_pretrain
    mlp_pretrain = args.mlp_pretrain
            
    topK = 10
    evaluation_threads = 1#mp.cpu_count()
    if mode=='train':
        print("NeuMF arguments: %s " % (args))
        model_out_file = 'Pretrain/%s_NeuMF_%d_%s_%d.h5' % (args.dataset, mf_dim, args.layers, time())

        # Loading data
        t1 = time()
        dataset = Dataset(args.path + args.dataset)
        train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
        num_users, num_items = train.shape
        #num_items = 65127
        print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
              % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))
        # Build model

        model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)
        if learner.lower() == "adagrad":
            model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
        elif learner.lower() == "rmsprop":
            model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
        elif learner.lower() == "adam":
            model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
        else:
            model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')

        # model = load_model('Pretrain/savedmodel.h5')
        # Load pretrain model
        if mf_pretrain != '' and mlp_pretrain != '':
            gmf_model = GMF.get_model(num_users, num_items, mf_dim)
            gmf_model.load_weights(mf_pretrain)
            mlp_model = MLP.get_model(num_users, num_items, layers, reg_layers)
            mlp_model.load_weights(mlp_pretrain)
            model = load_pretrain_model(model, gmf_model, mlp_model, len(layers))
            print("Load pretrained GMF (%s) and MLP (%s) models done. " % (mf_pretrain, mlp_pretrain))

        # Init performance
        (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
        best_hr, best_ndcg, best_iter = hr, ndcg, -1
        if args.out > 0:
            model.save_weights(model_out_file, overwrite=True)

        for epoch in xrange(num_epochs):
            t1 = time()
            # Generate training instances
            user_input, item_input, labels = get_train_instances(train, num_negatives)

            # Training
            hist = model.fit([np.array(user_input), np.array(item_input)],  # input
                             np.array(labels),  # labels
                             batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
            t2 = time()

            # Evaluation
            if epoch % verbose == 0:
                (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
                hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
                print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                      % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
                if hr > best_hr:
                    best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                    if args.out > 0:
                        model.save_weights(model_out_file, overwrite=True)
                        # model.save('Pretrain/savedmodel', overwrite=True)

        print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
        if args.out > 0:
            print("The best NeuMF model is saved to %s" % (model_out_file))
    if mode=='predict':
        model = get_model(6002, 65134, 8, [64, 32, 16, 8], [0, 0, 0, 0], 0.0)
        movie = 'Data/movies.dat'
        movies = []
        with open(movie, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("::")
                movies.append(int(arr[0]))
                line = f.readline()

        if learner.lower() == "adagrad":
            model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
        elif learner.lower() == "rmsprop":
            model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
        elif learner.lower() == "adam":
            model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy')
        else:
            model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
        num_users = 6002
        num_items = 65134
        t1 = time()
        #model.load_weights('Pretrain/ml-1m_NeuMF_8_[64,32,16,8]_1511548120.h5')
        model.load_weights('Pretrain/10m_NeuMF_8_[64,32,16,8]_1511676171.h5')
        t2 = time()
        print(t2 - t1)
        # Training model
        train = load_rating_file_as_matrix('/Applications/MAMP/htdocs/movie_recommendation/resources/test.csv')
        item_1 = load_rating_file_as_list('/Applications/MAMP/htdocs/movie_recommendation/resources/test.csv')
        userid = 6001
        user_input, item_input, labels = get_train_instances(train, 4)
        for i in range(20):
            model.fit([np.array(user_input), np.array(item_input)],  # input
                  np.array(labels),  # labels
                  batch_size=256, nb_epoch=1, verbose=0, shuffle=True)
        #itemlist = Unseen.getList(65132, item_1)
        itemlist = getUnseen(movies, item_1)
        # print(itemlist)
        items = np.array(itemlist)
        #print(items)
        users = np.full(len(items), userid, dtype="int32")
        predictions = model.predict([users, items], batch_size=100, verbose=0)
        map = {}
        #[150, 260, 296, 318, 356, 357, 364, 367, 380, 457]
        for i in xrange(len(items)):
            item = items[i]
            map[item] = predictions[i]
        ranklist = heapq.nlargest(15, map, key=map.get)
        with open("/Applications/MAMP/htdocs/movie_recommendation/resources/rank.txt", "w") as f:
            temp = []
            for i in range(10):
                temp.append(ranklist[i+4])
            f.write(str(temp))
        t3 = time()
        print(t3 - t2)
        #print(temp)
        #print(ranklist)
