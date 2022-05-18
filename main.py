import os
from flask import Flask, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import fasttext
import pickle
from firebase_admin import credentials, auth, initialize_app

app = Flask(__name__)
CORS(app)

cred = credentials.ApplicationDefault()
initialize_app(cred, options={"projectId": "scrapoo-eileen-new"})

vector_df = pd.read_csv("vector_df.tsv", sep="\t")
ft = fasttext.load_model("fasttext50.bin")
with open("random_forest_classification.model", "rb") as f:
    model = pickle.load(f)

@app.route('/predict')
def predict():
    id_token = request.headers["Authorization"].split()[-1]
    decoded_token = auth.verify_id_token(id_token)

    target_title = request.args.get("title")
    if target_title is None:
        return {"result": "invalid params"}, 500
    df = target_vec = vector_df[vector_df.title == target_title]
    if len(df):
        y_target = np.array(df.iloc[:, 3:])
        pred = model.predict(y_target)[0]
        idx = df.index.min()
        y_true = int(df.label_id[idx])
        label = df.label[idx]
    else:
        y_target = np.array([ft.get_word_vector(target_title)])
        pred = model.predict(y_target)[0]
        y_true = 1000
        label = vector_df[vector_df.label_id == pred].label.min()
    result = {
        "result": {
            "y_true": int(y_true),
            "label": label,
            "pred": int(pred),
            "result": bool(y_true == pred)
        }
    }
    return result, 200



@app.route('/recommend_items')
def recommend_items():
    id_token = request.headers["Authorization"].split()[-1]
    decoded_token = auth.verify_id_token(id_token)

    target_title = request.args.get("title")
    if target_title is None:
        return {"result": "invalid params"}, 500
    target_vec = vector_df[vector_df.title == target_title].iloc[:, 3:]
    other_vec = vector_df[vector_df.title != target_title].iloc[:, 3:]
    if not len(target_vec):
        target_vec = np.array([ft.get_word_vector(target_title)])

    cos = cosine_similarity(target_vec, other_vec)[0]
    indices = np.argsort(cos)+1
    recommend_ids = np.concatenate([indices[-5:], indices[:5]])
    titles = list(vector_df.loc[recommend_ids].title)
    return {"result": titles}, 200

@app.route('/')
def index():
    return 'Hello Eileen'

# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
