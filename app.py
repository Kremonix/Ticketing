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
    chosen_category = None
    predicted_category = None
    ticket_description = None
    mismatch = False
    success = False

    if request.method == "POST":
        chosen_category = request.form.get("chosen_category")
        ticket_description = request.form.get("ticket_description", "").strip()

        # Beschreibung mit TF-IDF vektorisieren
        text_vector = vectorizer.transform([ticket_description])
        predicted_category = svm_model.predict(text_vector)[0]

        # Kategorien vergleichen
        if predicted_category == chosen_category:
            success = True
        else:
            mismatch = True

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
    if request.method == "POST":
        # Liste aller Formularfelder (basierend auf `survey.html`)
        fields = [
            "experience",
            "intuitive",
            "prediction_effectiveness",
            "prediction_inaccuracy",
            "category_accuracy",
            "problem_description",
            "missing_fields",
            "missing_fields_details",
            "submission_time",
            "response_time",
            "enhancements",
            "enhancements_details",
            "additional_features",
            "reuse_likelihood",
            "necessary_information",
            "necessary_information_details",
            "additional_comments"
        ]

        # Erfasse alle Eingaben aus dem Formular
        survey_data = {field: request.form.get(field, "") for field in fields}

        # Verbesserungen (Checkboxes) als Liste erfassen
        survey_data["enhancements"] = request.form.getlist("enhancements")

        # CSV-Datei öffnen oder erstellen
        file_exists = os.path.isfile("survey_results.csv")
        with open("survey_results.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header schreiben, falls die Datei neu ist
            if not file_exists:
                writer.writerow(fields)

            # Daten schreiben
            writer.writerow([survey_data[field] for field in fields])

        # Nach dem Speichern -> Danke-Seite anzeigen
        return render_template("thank_you.html")

    return render_template("survey.html")

if __name__ == "__main__":
    app.run(debug=True)
