import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# 1. Datensatz laden
file_path = "all_tickets_processed_improved_v3.csv"  # Passe den Pfad an
df = pd.read_csv(file_path)

# 2. Text- und Zielspalten definieren
X = df["Document"]
y = df["Topic_group"]

# 3. Textdaten in numerische Features umwandeln (TF-IDF)
vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(X)

# 4. 80/20 Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# 5. Modelltraining mit SVM
svm_model = LinearSVC(max_iter=1000, random_state=42)
svm_model.fit(X_train, y_train)

# 6. Vorhersage auf Testdaten
y_pred = svm_model.predict(X_test)

# 7. Modellbewertung
print("\nModellbewertung:")
print(classification_report(y_test, y_pred))

joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# 8. Neue Tickets testen
new_tickets = [
    "I cannot connect to the network, please help.",
    "Please reset my account password.",
    "The device is not working as expected.",
    "How can I access the company portal from home?"
]

# Neue Tickets vektorisieren und vorhersagen
new_tickets_transformed = vectorizer.transform(new_tickets)
predictions = svm_model.predict(new_tickets_transformed)

# Ergebnisse anzeigen
print("\nVorhersagen für neue Tickets:")
for ticket, prediction in zip(new_tickets, predictions):
    print(f"Beschreibung: {ticket} -> Vorhergesagte Kategorie: {prediction}")
