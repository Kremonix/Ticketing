from flask import Flask, render_template, request, redirect, url_for
import joblib
import csv
import os
from datetime import datetime

app = Flask(__name__)

# Load the trained model and vectorizer
svm_model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Categories
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

# CSV-Dateipfad
CSV_FILE = "simulation_results.csv"

@app.route("/")
def home():
    return redirect(url_for("intro"))

@app.route("/intro", methods=["GET"])
def intro():
    return render_template("intro.html")

@app.route("/tickets", methods=["GET", "POST"])
def index():
    chosen_category = None
    predicted_category = None
    ticket_description = None
    mismatch = False
    success = False

    if request.method == "POST":
        if request.form.get("change_category"):
            chosen_category = request.form.get("chosen_category")
            ticket_description = request.form.get("ticket_description")
            success = True  
        else:
            chosen_category = request.form.get("chosen_category")
            ticket_description = request.form.get("ticket_description", "").strip()
            text_vector = vectorizer.transform([ticket_description])
            predicted_category = svm_model.predict(text_vector)[0]

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

@app.route("/thank_you", methods=["GET"])
def thank_you():
    """
    Speichert das Ticket erst, wenn der Nutzer auf "Finish Simulation" klickt.
    """
    ticket_description = request.args.get("ticket_description", "")
    chosen_category = request.args.get("chosen_category", "")
    predicted_category = request.args.get("predicted_category", "")

    if ticket_description and chosen_category:
        save_ticket_to_csv(ticket_description, chosen_category, predicted_category)

    return render_template("thank_you.html")

def save_ticket_to_csv(ticket_description, chosen_category, predicted_category):
    file_exists = os.path.isfile(CSV_FILE)
    
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Ticket Description", "Chosen Category", "Predicted Category"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ticket_description,
            chosen_category,
            predicted_category
        ])

if __name__ == "__main__":
    app.run(debug=True)
