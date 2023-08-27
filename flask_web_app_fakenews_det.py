from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import string





app = Flask(__name__  , template_folder='html')
#make the vectorizer
vect=TfidfVectorizer(stop_words='english', max_df=0.7)
#load the saved classifier model 
loaded_model = pickle.load(open('model', 'rb'))
#open the training dataframe
dataframe = pd.read_csv('train.csv')
#drop the duplicates and missing values
dataframe.drop_duplicates(inplace=True)
dataframe.dropna(axis=0, inplace=True)
x = dataframe['title']+ '' +dataframe['author']
y = dataframe['label']
#split tha data
x_train, x_test,y_train, y_test = train_test_split(x, dataframe['label'], test_size=0.20, random_state=0)


def fake_news_det(news):
    news=str(news)
    tfid_x_train = vect.fit_transform(x_train)
    tfid_x_test = vect.transform(x_test)
    input_data = [news]
    vectorized_input_data = vect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

@app.route('/', methods=['POST' , 'GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST' , 'GET'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        print(pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)