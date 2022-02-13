import json
import pymongo
import pandas as pd 
from flask import Flask, flash, request, redirect, jsonify
from flask_swagger_ui import get_swaggerui_blueprint
from werkzeug.utils import secure_filename
from celery import Celery
from bson.objectid import ObjectId

app = Flask(__name__)

SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "MLProvider"
    }
)
app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)

celery_cluster = Celery('tasks',
                    broker='amqp://admin:mypass@rabbitmq_host:5672',
                    backend='mongodb://mongodb:27017/MLProvider-celery')

d = {'CELERY_ACCEPT_CONTENT' : ['pickle'],
'CELERY_TASK_SERIALIZER' : 'pickle',
'CELERY_RESULT_SERIALIZER' : 'pickle'}

celery_cluster.conf.update(d)

url = 'mongodb://mongodb:27017/?authSource=admin'
mongo_client = pymongo.MongoClient(url)

ROOT = '/api/v1/'
ALLWOED_DATASET_FORMAT = 'csv'

@app.route(ROOT+'dataset', methods=['POST','GET'])
def post_dataset():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'status': 400, 'msg': 'No file part'})

        file = request.files['file']
        label_col = request.form['label_column']
        if file.filename == '':
            return jsonify({'status': 400, 'msg': 'No selected part'})
        if file:
            filename = secure_filename(file.filename)
            if filename.rsplit('.', 1)[1].lower() != ALLWOED_DATASET_FORMAT:
                flash('Bad format. not csv!')
                return redirect(request.url)
            mongo_db = mongo_client["MLProvider"]
            collection = mongo_db['datasets']
            df = pd.read_csv(file)
            x = collection.insert_one({'data' : df.to_json(), 'label_column': label_col})
            return jsonify({'status': 200, 'dataset_id': str(x.inserted_id)})
        return jsonify({'status': 500})

@app.route(ROOT+'/model', methods=['POST', 'GET'])
def model():
    if request.method == 'GET':
        return jsonify({'status': 200, "model_typs": ['LogisticRegression', 'RandomForestClassifier', 'AdaBoostClassifier']})
    if request.method == 'POST':
        app.logger.info("Invoking traning ")
        data = json.loads(request.data)
        if 'model_type' not in data:
            flash('model type missing.')
            return redirect(request.url)
        if 'dataset_id' not in data:
            flash('dataset_id type missing.')
            return redirect(request.url)
        r = celery_cluster.send_task('tasks.train_model', kwargs=data)
        app.logger.info(r.backend)
        return jsonify({'status': 200, 'model_id':r.id})

@app.route(ROOT+'/model/<model_id>')
def get_status(model_id):
    status = celery_cluster.AsyncResult(model_id, app=celery_cluster)
    return "Status of the Task " + str(status.state)

@app.route(ROOT+'/prediction', methods=['POST'])
def task_result():
    if request.method == 'POST':
        data = json.loads(request.data)
        if 'model_id' not in data:
             jsonify({'status': 400, 'msh': "no model id."})
        if 'data' not in data:
             jsonify({'status': 400, 'msh': "no data."})
        model = celery_cluster.AsyncResult(data['model_id']).result
        res = model.predict([data['data']])
        return jsonify({'status': 200, 'results' : res.tolist()[0]})




