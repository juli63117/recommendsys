from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import json
from pathlib import Path

app = Flask(__name__)
CORS(app)
BASE_DIR = Path(__file__).resolve().parent

# API-роут
@app.route("/api/recommendations")
def get_recommendations():
    path = BASE_DIR / "recommender" / "output" / "recommendations.json"
    print(f"Путь к файлу: {path}")
    
    if not path.exists():
        print("ОШИБКА: Файл не найден!")
        return jsonify({"error": "File not found"}), 404
    
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"Прочитанные данные: {data}")
            return jsonify(data)
    except Exception as e:
        print(f"ОШИБКА ЧТЕНИЯ: {e}")
        return jsonify({"error": "Invalid JSON"}), 500

if __name__ == "__main__":
    app.run(debug=True)