from flask import Flask, render_template, request, redirect, url_for
import joblib
import csv
import os

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

@app.route("/")
def home():
    """
    Redirects to the intro page.
    """
    return redirect(url_for("intro"))

@app.route("/intro", methods=["GET"])
def intro():
    """
    Displays the introduction page for the experiment.
    """
    return render_template("intro.html")

@app.route("/tickets", methods=["GET", "POST"])
def index():
    """
    Handles ticket submission and category evaluation.
    """
    chosen_category = None
    predicted_category = None
    ticket_description = None
    mismatch = False
    success = False

    if request.method == "POST":
        # Check if the user clicked the "Change" button
        if request.form.get("change_category"):
            # Update chosen_category to the predicted one
            chosen_category = request.form.get("chosen_category")
            ticket_description = request.form.get("ticket_description")
            success = True  # Assume success since the suggested category is now chosen
        else:
            # Regular submission logic
            chosen_category = request.form.get("chosen_category")
            ticket_description = request.form.get("ticket_description", "").strip()

            # Vectorize the description and predict the category
            text_vector = vectorizer.transform([ticket_description])
            predicted_category = svm_model.predict(text_vector)[0]

            # Check for mismatch
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
    """
    Displays a survey page (GET) and saves the results (POST).
    """
    if request.method == "POST":
        # List of all form fields in survey.html
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

        # Capture all inputs from the form
        survey_data = {field: request.form.get(field, "") for field in fields}

        # Handle multiple selections for enhancements
        survey_data["enhancements"] = request.form.getlist("enhancements")

        # Save data to CSV
        file_exists = os.path.isfile("survey_results.csv")
        with open("survey_results.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write header if file is new
            if not file_exists:
                writer.writerow(fields)

            # Write data row
            writer.writerow([survey_data[field] for field in fields])

        # Redirect to thank you page after submission
        return render_template("thank_you.html")

    return render_template("survey.html")


if __name__ == "__main__":
    app.run(debug=True)
