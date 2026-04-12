from flask import Flask, request, jsonify
from flask_cors import CORS
import recommendation_model as rm
import os

app = Flask(__name__)
CORS(app)

# Initialize the recommendation system
print("Initializing Recommendation System...")
RULES = rm.setup_system('simple_groceries.csv')
print("System ready!")

@app.route('/recommend', methods=['POST'])
def recommend_endpoint():
    try:
        data = request.get_json()
        user_input = data.get('item', '')
        
        if not user_input:
            return jsonify({"error": "No input provided"}), 400
            
        recs = rm.recommend(user_input, RULES, k=5)
        
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

@app.route('/top-products', methods=['GET'])
def top_products():
    try:
        top = rm.get_top_products(10)
        return jsonify(top)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/model-stats', methods=['GET'])
def model_stats():
    try:
        stats = rm.get_model_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/top-combinations', methods=['GET'])
def top_combinations():
    try:
        combos = rm.get_top_combinations(5)
        return jsonify(combos)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/smart-insights', methods=['GET'])
def smart_insights():
    try:
        insights = rm.get_smart_insights()
        return jsonify(insights)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "online", "message": "Backend is running!"})

if __name__ == '__main__':
    # Use port 8081 as before
    app.run(host='0.0.0.0', port=8081, debug=True)
