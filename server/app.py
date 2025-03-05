from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# Temporary storage for emails (use a database in production)
emails = []

@app.route('/store-email', methods=['POST'])
def store_email():
    data = request.json
    email = data.get("email")
    
    if email:
        emails.append(email)  # Store email
        print("Stored Emails:", emails)  # Debugging print statement
        return jsonify({"message": "Email stored successfully"}), 200
    
    return jsonify({"error": "Invalid email"}), 400

if __name__ == '__main__':
    app.run(debug=True)
