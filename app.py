from flask import Flask
from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy 
from keras.datasets import imdb 
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import LSTM 
from keras.layers.embeddings import Embedding 
from keras.preprocessing import sequence

app = Flask(__name__,template_folder='templates')

top_words = 5000 
max_review_length = 500 
# create the model 
embedding_vector_length = 32 
model = Sequential() 
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length)) 
model.add(LSTM(100)) 
#model.add(Flatten()) 
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy']) 


# load weights into new model

model.load_weights("model.h5")
print("Loaded Model from disk")
import keras
NUM_WORDS=1000 # only use top 1000 words
INDEX_FROM=3   # word index offset
word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

id_to_word = {value:key for key,value in word_to_id.items()}

#print(word_to_id["i"])
from numpy import array

@app.route("/")
def home():
    #return "Hello, World!"
    return render_template('index.html')
@app.route("/predict")
def predict():
    bad = "this movie was terrible and bad"
    good = "i really liked the movie and had fun"
    for review in [good]:
        tmp = []
        for word in review.split(" "):
            tmp.append(word_to_id[word])
            tmp_padded = sequence.pad_sequences([tmp], maxlen=max_review_length) 
            #print("%s . Sentiment: %s" % (review,model.predict(array([tmp_padded][0]))[0][0]))
	#return "Hello, World!"
    return 'You entered: {}'.format("%s . Sentiment: %s" % (review,model.predict(array([tmp_padded][0]))[0][0]))
@app.route("/test",methods=['POST'])
def test():
    if request.method == 'POST':
        namequery=request.form['namequery']
        data=[namequery]
		
        for review in data:
            tmp = []
            for word in review.split(" "):
                if (word_to_id[word]!=''):
                    tmp.append(word_to_id[word])
                    tmp_padded = sequence.pad_sequences([tmp], maxlen=max_review_length) 
                my_pred=model.predict(array([tmp_padded][0]))[0][0]
                if (my_pred>=0.6):
                    rev='good'
                else:
                    rev='bad'
    
    return render_template('index.html',prediction=rev)
    

    
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,threaded=False)