from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# Temporary storage for emails (use a database in production)
emails = []
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route('/store-email', methods=['POST'])
def store_email():
    data = request.json
    email = data.get("email")
    
    if email:
        emails.append(email)  # Store email
        print("Stored Emails:", emails)  # Debugging print statement
        return jsonify({"message": "Email stored successfully"}), 200
    
    return jsonify({"error": "Invalid email"}), 400

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"message": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"message": "No selected file"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    return jsonify({"message": "File uploaded successfully", "filename": file.filename}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
