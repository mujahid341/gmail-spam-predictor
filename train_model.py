import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Rename important columns
df = df.rename(columns={'v1': 'label', 'v2': 'message'})

# Keep only the required columns
df = df[['label', 'message']]

# Drop missing and duplicate rows
df = df.dropna()
df = df.drop_duplicates()

# Convert text to lowercase
df['label'] = df['label'].str.lower()
df['message'] = df['message'].str.lower()

# Map labels to numerical values: ham = 0, spam = 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Convert text messages to numeric vectors
vectorizer = CountVectorizer(stop_words='english')  # remove common words like the , and etc
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
