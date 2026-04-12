from flask import Flask, request, jsonify
from flask_cors import CORS
import recommendation_model as rm
import os

app = Flask(__name__)
CORS(app)

# Initialize the recommendation system
print("Initializing Hybrid Recommendation System...")
RULES = rm.setup_system('simple_groceries.csv')
print("System ready!")

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        user_input = data.get('item', '')
        
        if not user_input:
            return jsonify({"error": "No input provided"}), 400
            
        recs = rm.hybrid_recommend(user_input, RULES, k=5)
        
        # Add some metadata for the "Insights" section
        metadata = {
            "total_rules": len(RULES),
            "avg_confidence": round(RULES['confidence'].mean(), 2) if not RULES.empty else 0,
            "avg_lift": round(RULES['lift'].mean(), 2) if not RULES.empty else 0
        }
        
        return jsonify({
            "recommendations": recs,
            "metadata": metadata
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "online", "message": "Backend is running!"})

if __name__ == '__main__':
    # Use port 8080 as before
    app.run(host='0.0.0.0', port=8080, debug=True)
