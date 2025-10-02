from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Sample news articles
texts = [
    "NASA confirms water on Mars",                      # Real
    "Aliens built the pyramids, scientists say",        # Fake
    "Government launches new healthcare plan",          # Real
    "Celebrity cloned in secret lab",                   # Fake
    "New vaccine approved by WHO",                      # Real
    "Time traveler spotted in 1920s photo",             # Fake
]

# Corresponding labels (0 = Real, 1 = Fake)
labels = [0, 1, 0, 1, 0, 1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save model and vectorizer
joblib.dump(model, 'models/model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
