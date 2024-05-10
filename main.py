
from tensorflow import keras
from fastapi import FastAPI
import uvicorn
from keras.layers import TextVectorization
import pickle

app= FastAPI()
from_disk = pickle.load(open("tv_layer.pkl", "rb"))
vectorizer = TextVectorization(max_tokens=from_disk['config']['max_tokens'],
                                          output_mode='int',
                                          output_sequence_length=from_disk['config']['output_sequence_length'])

vectorizer.set_weights(from_disk['weights'])

model= keras.models.load_model("toxic_classifier.h5",compile=False)
pakora=["1","2","toxic","severe_toxic","obscene","threat","insult","identity_hate"]

@app.get("/ping")
async def ping():
    return "server is alive"


def predict(comment):
    vectorized=vectorizer([comment])
    res=model.predict(vectorized,0)
    text=''
    for idx,col in enumerate(pakora[2:-1]):
        text+='{}: {} '.format(col, res[0][idx]>0.5)
    return text

@app.post("/find")
async def find(
        comment : str):
    result = predict(comment)
    return{
        'Result' : result
    }
    pass

if __name__=="__main__":
    uvicorn.run(app,host='localhost',port=8000)
