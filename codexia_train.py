# import library
import os
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from utils import JSONParser, Preprocessor

# load data
path = "dataset/data.json"

pcsr = Preprocessor()
jp = JSONParser()
jp.parse(path)
df = jp.get_dataframe()

# praproses data
df['text_input_prep'] = df.text_input.apply(pcsr.preprocess)

# pemodelan
pipeline = make_pipeline(CountVectorizer(),
                         MultinomialNB())

# train
print("[INFO] Training Data ...")
pipeline.fit(df.text_input_prep, df.intents)

# save model
model_path = os.path.join(os.path.join(os.getcwd(), 'model'), "model_chatbot.pkl")
with open(model_path, "wb") as model_file:
    pickle.dump(pipeline, model_file)

print(f"[INFO] Model saved to {model_path}")