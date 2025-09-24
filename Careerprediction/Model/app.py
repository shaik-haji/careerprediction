from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd

app = Flask(__name__, template_folder=r"D:\Career\Careerprediction\templates")

# Load the trained model
model = pickle.load(open(r'D:\Career\Careerprediction\Model\model.pkl', 'rb'))
label_encoders = pickle.load(open(r'D:\Career\Careerprediction\Model\label_encoders.pkl', 'rb'))

le = label_encoders["Career"]
career_classes = dict(zip(le.transform(le.classes_), le.classes_))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data directly as integers/floats
        data = request.form
        print("📥 Received Input Data:", data)  # Debugging

        encoded_data = {
            "GPA": float(data["GPA"]),
            "Internships": int(data["Internships"]),
            "Projects": int(data["Projects"]),
            "Coding_Skills": int(data["Coding_Skills"]),
            "Communication_Skills": int(data["Communication_Skills"]),
            "Leadership_Experience": int(data["Leadership_Experience"]),
            "Extracurricular_Activities": int(data["Extracurricular_Activities"]),
            "Preferred_Work_Environment": int(data["Preferred_Work_Environment"])
        }

        print("🛠️ Encoded Data for Model:", encoded_data)  # Debugging

        # Convert into DataFrame for model prediction
        df = pd.DataFrame([encoded_data])
        print(df)
        # Make prediction
        print(model.predict(df)[0])
        prediction = model.predict(df)[0]

        career = ""
        if prediction in career_classes:
            career = career_classes[prediction]

        # Render prediction result page
        return render_template("predict.html", career=career)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
