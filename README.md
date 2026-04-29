#Fake News Detection System

A Machine Learning web application that detects whether a news article is **REAL** or **FAKE** using Python and Flask.

---

## Project Overview

This project builds an end-to-end Fake News Detection System that:
- Trains a machine learning model on 44,898 real and fake news articles
- Achieves **99.43% accuracy** on test data
- Provides a clean web interface where users paste any news article and get an instant prediction

---

## Demo

1. Run the app: `python app.py`
2. Open browser: `http://127.0.0.1:5000`
3. Paste a news article and click **Analyze News**
4. Get result: **REAL** or **FAKE**

---

## Project Structure

```
fake_news_detector/
│
├── templates/
│   └── index.html        # Web page (HTML + CSS)
│
├── venv/                 # Virtual environment
│
├── app.py                # Flask web application
├── preprocess.py         # Data cleaning script
├── train_model.py        # Model training script
├── test_setup.py         # Setup check script
│
├── model.pkl             # Saved trained model
├── tfidf.pkl             # Saved text vectorizer
├── cleaned_data.csv      # Processed dataset
│
├── Fake.csv              # Raw fake news data
├── True.csv              # Raw real news data
│
└── README.md             # Project documentation
```

---

## Tools and Libraries

| Tool | Purpose |
|---|---|
| Python 3.12 | Core programming language |
| pandas | Data loading and processing |
| scikit learn | Machine learning model |
| Flask | Web application framework |
| joblib | Save and load the trained model |
| HTML and CSS | Frontend web interface |

---

## Dataset

- **Source:** Kaggle — Fake and Real News Dataset
- **Fake news articles:** 23,481
- **Real news articles:** 21,417
- **Total articles:** 44,898
- **Train and Test split:** 80% training, 20% testing

---

## How It Works

```
Step 1 — Load Data
   Fake.csv + True.csv (44,898 articles)

Step 2 — Clean Data
   Label fake as 0 and real as 1
   Merge title and text together
   Remove empty rows and shuffle

Step 3 — Extract Features
   Convert text to numbers
   50,000 features per article

Step 4 — Train Model
   Passive Aggressive Classifier
   Trained on 35,918 articles

Step 5 — Web App
   User pastes news article
   Model predicts REAL or FAKE
```

---

## How to Run

### Step 1 — Activate the environment

**Windows:**
```
venv\Scripts\activate
```

**Mac or Linux:**
```
source venv/bin/activate
```

### Step 2 — Install libraries

```
pip install flask scikit-learn pandas numpy nltk joblib
```

### Step 3 — Process the data

```
python preprocess.py
```

### Step 4 — Train the model

```
python train_model.py
```

### Step 5 — Start the web app

```
python app.py
```

### Step 6 — Open in browser

```
http://127.0.0.1:5000
```

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 99.43% |
| Precision | 99.4% |
| Recall | 99.3% |
| F1 Score | 99.4% |
| Training Samples | 35,918 |
| Testing Samples | 8,980 |

---

## Sample Test Cases

**Fake News Input:**
```
SHOCKING: NASA confirms aliens have landed in New Mexico.
The government has been hiding this secret for decades.
Share this before it gets deleted!
```
**Result: FAKE**

**Real News Input:**
```
The Federal Reserve raised interest rates by 25 basis points on Wednesday.
Fed Chair said the central bank remains committed to bringing inflation
back down to its 2 percent target.
```
**Result: REAL**

---

## Future Improvements

- Use deep learning models for higher accuracy
- Add live news checking via a news API
- Build a browser extension
- Deploy to a cloud server for public access

---

## Author

**Sahana Hiremath**
College Project — Fake News Detection System
April 2026

---

## Credits

- Dataset from Kaggle by Clement Bisaillon
- Built using Python, Flask, and scikit learn
