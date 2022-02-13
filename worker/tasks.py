from cProfile import label
import time
import pymongo
import pandas as pd
from celery import Celery
from celery.utils.log import get_task_logger
from bson.objectid import ObjectId
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
import pickle as pkl
import numpy as np
from bson.binary import Binary

clfs = {
    'LogisticRegression': LogisticRegression,
    'RandomForestClassifier': RandomForestClassifier,
    'AdaBoostClassifier': AdaBoostClassifier
}
logger = get_task_logger(__name__)

app = Celery('tasks',
             broker='amqp://admin:mypass@rabbitmq_host:5672',
             backend='mongodb://mongodb:27017/MLProvider-celery')

d = {'CELERY_ACCEPT_CONTENT' : ['pickle'],
'CELERY_TASK_SERIALIZER' : 'pickle',
'CELERY_RESULT_SERIALIZER' : 'pickle'}
app.conf.update(d)
@app.task()
def train_model(model_type: str, dataset_id: str):
    logger.info('Geting dataset')
    url ='mongodb://mongodb:27017/?authSource=admin'
    mongo_client = pymongo.MongoClient(url)
    mongo_db = mongo_client['MLProvider']
    collection = mongo_db['datasets']
    dataset = collection.find_one({ "_id": ObjectId(dataset_id)})
    label_column = dataset['label_column']
    df = pd.read_json(dataset['data'])
    df = df.replace('?', np.NaN)
    train_y = df[[label_column]]
    train_x = df.drop(columns=[label_column])
    del df
    imputer = KNNImputer(n_neighbors=1)
    imputer.fit(train_x)
    train_x = imputer.transform(train_x)
    logger.info('dataset ready')
    if model_type not in clfs:
        logger.error('invalid model type')
        return "invalid model type"
    clf = clfs[model_type]()
    clf.fit(train_x,train_y)
    logger.info('train Finished ')
    picked_data = pkl.dumps(clf)
    logger.info('model converted to pickel ')
    return clf
