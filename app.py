from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Das zuvor trainierte Modell und den Vectorizer laden
# Achte darauf, dass deine Modelle im selben Verzeichnis liegen wie diese app.py
# und dass die Dateinamen korrekt sind.
svm_model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Die von dir gewünschten Kategorien
CATEGORIES = [
    "Hardware",
    "HR Support",
    "Access",
    "Miscellaneous",
    "Storage",
    "Purchase",
    "Internal Project",
    "Administrative rights"
]

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Zeigt ein Formular zum Einreichen eines Tickets.
    Beim POST wird der Text klassifiziert und mit der
    ausgewählten Kategorie verglichen.
    """
    chosen_category = None
    predicted_category = None
    ticket_description = None
    mismatch = False

    if request.method == "POST":
        # Formulardaten auslesen
        chosen_category = request.form.get("chosen_category")
        ticket_description = request.form.get("ticket_description")

        # Beschreibung mit TF-IDF vektorisieren
        text_vector = vectorizer.transform([ticket_description])
        # Vorhersage treffen
        predicted_category = svm_model.predict(text_vector)[0]

        # Vergleichen, ob Modell-Kategorie und gewählte Kategorie übereinstimmen
        if predicted_category != chosen_category:
            mismatch = True

    return render_template(
        "index.html",
        categories=CATEGORIES,
        mismatch=mismatch,
        chosen_category=chosen_category,
        predicted_category=predicted_category,
        ticket_description=ticket_description
    )

if __name__ == "__main__":
    # Starte die Flask-App (Standard-Port 5000)
    app.run(debug=True)
