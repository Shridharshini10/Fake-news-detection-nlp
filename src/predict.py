import joblib

model = joblib.load('models/model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

def predict_news(news_text):
    transformed_text = vectorizer.transform([news_text])
    prediction = model.predict(transformed_text)
    return "Fake News" if prediction[0] == 1 else "Real News"

if __name__ == "__main__":
    news = input("Enter the news article text:\n")
    result = predict_news(news)
    print(f"\nPrediction: {result}")
