from flask import Flask, render_template, request, redirect, url_for
import joblib
import csv
import os

app = Flask(__name__)

# Das zuvor trainierte Modell und den Vectorizer laden
svm_model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Kategorien
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
    success = False  # Erfolgsmeldung

    if request.method == "POST":
        chosen_category = request.form.get("chosen_category")
        ticket_description = request.form.get("ticket_description", "").strip()

        # Beschreibung mit TF-IDF vektorisieren
        text_vector = vectorizer.transform([ticket_description])
        # Vorhersage treffen
        predicted_category = svm_model.predict(text_vector)[0]

        # Kategorien vergleichen
        if predicted_category == chosen_category:
            success = True  # Erfolgsmeldung anzeigen
        else:
            mismatch = True  # Fehlermeldung anzeigen

    return render_template(
        "index.html",
        categories=CATEGORIES,
        mismatch=mismatch,
        success=success,
        chosen_category=chosen_category,
        predicted_category=predicted_category,
        ticket_description=ticket_description
    )

@app.route("/survey", methods=["GET", "POST"])
def survey():
    """
    Zeigt eine Umfrage-Seite (GET) und speichert die Ergebnisse (POST).
    """
    if request.method == "POST":
        # Felder aus dem Survey-Formular
        user_name = request.form.get("user_name", "")
        satisfaction = request.form.get("satisfaction", "")
        comments = request.form.get("comments", "")

        # Survey-Daten speichern
        file_exists = os.path.isfile("survey_results.csv")
        with open("survey_results.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["user_name", "satisfaction", "comments"])
            writer.writerow([user_name, satisfaction, comments])

        # Nach dem Speichern -> "thank_you.html"
        return render_template("thank_you.html", user_name=user_name)

    # GET-Request: Zeige das Umfrage-Formular
    return render_template("survey.html")

if __name__ == "__main__":
    app.run(debug=True)
