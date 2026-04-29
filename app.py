from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the saved model and vectorizer
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

@app.route("/")
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form["news_text"]
    
    if not news_text.strip():
        return render_template("index.html", 
                             prediction="Please enter some text!",
                             news_text=news_text)
    
    # Transform input using TF-IDF
    transformed = tfidf.transform([news_text])
    
    # Make prediction
    prediction = model.predict(transformed)[0]
    
    if prediction == 1:
        result = "REAL"
    else:
        result = "FAKE"
    
    return render_template("index.html",
                         prediction=result,
                         news_text=news_text)

if __name__ == "__main__":
    app.run(debug=True)