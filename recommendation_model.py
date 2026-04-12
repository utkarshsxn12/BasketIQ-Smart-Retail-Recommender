import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from sklearn.metrics.pairwise import cosine_similarity
import os
import difflib

# 🔹 Product Synonyms & Brand Mapping
PRODUCT_SYNONYMS = {
    "colgate": "toothpaste",
    "pepsodent": "toothpaste",
    "sensodyne": "toothpaste",
    "maggi": "maggi",
    "yippee": "cup noodles",
    "knorr": "cup noodles",
    "oreo": "cookies",
    "hide and seek": "cookies",
    "parle g": "biscuits",
    "coke": "soft drink",
    "pepsi": "soft drink",
    "thums up": "soft drink",
    "sprite": "soft drink",
    "lays": "chips",
    "kurkure": "kurkure",
    "bingo": "chips",
    "doritos": "nachos",
    "amul": "butter",
    "mother dairy": "milk",
    "tide": "detergent",
    "surf excel": "detergent",
    "rin": "detergent",
    "dettol": "soap",
    "lifebuoy": "soap",
    "dove": "soap",
    "lux": "soap",
    "colin": "toilet cleaner",
    "harpic": "toilet cleaner",
    "lizol": "floor cleaner",
    "vim": "dishwash gel"
}

# 🔹 Product Categories Mapping
PRODUCT_CATEGORIES = {
    "milk": "dairy", "butter": "dairy", "cheese": "dairy", "yogurt": "dairy", "paneer": "dairy", "fresh cream": "dairy",
    "bread": "bakery", "brown bread": "bakery", "sandwich bread": "bakery", "jam": "bakery", "cake": "bakery", "muffins": "bakery",
    "tea": "beverages", "coffee": "beverages", "green tea": "beverages", "instant coffee": "beverages", "juice": "beverages", "fruit juice": "beverages", "soft drink": "beverages", "energy drink": "beverages", "mineral water": "beverages", "coconut water": "beverages", "soda": "beverages", "iced tea": "beverages",
    "biscuits": "snacks", "cookies": "snacks", "chips": "snacks", "nachos": "snacks", "popcorn": "snacks", "chocolate": "snacks", "bhujia": "snacks", "kurkure": "snacks", "mixture": "snacks", "roasted cashews": "snacks", "salted peanuts": "snacks",
    "eggs": "breakfast", "oats": "breakfast", "cornflakes": "breakfast", "honey": "breakfast", "muesli": "breakfast", "peanut butter": "breakfast", "maggi": "breakfast", "cup noodles": "breakfast",
    "apple": "fruits", "banana": "fruits", "orange": "fruits", "grapes": "fruits", "mango": "fruits", "watermelon": "fruits", "kiwi": "fruits", "papaya": "fruits", "pomegranate": "fruits", "guava": "fruits", "pineapple": "fruits", "pear": "fruits",
    "tomato": "vegetables", "onion": "vegetables", "potato": "vegetables", "cucumber": "vegetables", "spinach": "vegetables", "carrot": "vegetables", "peas": "vegetables", "capsicum": "vegetables", "broccoli": "vegetables", "cauliflower": "vegetables", "lady finger": "vegetables", "ginger": "vegetables", "garlic": "vegetables", "coriander": "vegetables", "green chili": "vegetables",
    "soap": "personal_care", "shampoo": "personal_care", "toothpaste": "personal_care", "toothbrush": "personal_care", "face wash": "personal_care", "hand wash": "personal_care", "body lotion": "personal_care", "deodorant": "personal_care", "shaving cream": "personal_care",
    "detergent": "household", "dishwash gel": "household", "toilet cleaner": "household", "floor cleaner": "household", "garbage bags": "household", "toilet paper": "household", "kitchen roll": "household", "napkins": "household",
    "rice": "staples", "wheat flour": "staples", "dal": "staples", "toor dal": "staples", "moong dal": "staples", "sugar": "staples", "salt": "staples", "cooking oil": "staples", "ghee": "staples", "mustard oil": "staples", "turmeric": "staples", "red chili powder": "staples", "jeera": "staples",
    "chicken": "protein", "mutton": "protein", "fish": "protein", "tofu": "protein", "soya chunks": "protein", "frozen peas": "protein", "sweet corn": "protein"
}

# Related Categories
RELATED_CATEGORIES = {
    "dairy": ["bakery", "breakfast"],
    "bakery": ["dairy", "beverages"],
    "beverages": ["snacks", "bakery"],
    "snacks": ["beverages"],
    "breakfast": ["dairy", "beverages"],
    "fruits": ["vegetables", "breakfast"],
    "vegetables": ["fruits", "staples"],
    "staples": ["vegetables", "protein"],
    "protein": ["vegetables", "staples"],
    "personal_care": ["household"],
    "household": ["personal_care"]
}

# Global similarity matrix and items
ITEM_SIM_DF = None
ITEM_FREQS = None

# 🔹 1. load_data(file_path)
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            items = [item.strip() for item in line.split(',') if item.strip()]
            if items: data.append(items)
    df = pd.DataFrame(data)
    print(f"Dataset loaded successfully from {file_path}")
    return df

# 🔹 3. create_transactions(df)
def create_transactions(df):
    transactions = []
    global ITEM_FREQS
    all_items = []
    for _, row in df.iterrows():
        items = []
        for item in row:
            if pd.notna(item):
                clean_item = str(item).lower().strip()
                if clean_item and clean_item not in ['nan', 'none']:
                    items.append(clean_item)
                    all_items.append(clean_item)
        if items: transactions.append(items)
    
    ITEM_FREQS = pd.Series(all_items).value_counts()
    print(f"Created {len(transactions)} cleaned transactions.")
    return transactions

# 🔹 4. encode_transactions(transactions)
def encode_transactions(transactions):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Calculate Cosine Similarity
    global ITEM_SIM_DF
    # Transpose so items are rows
    item_matrix = df_encoded.T.values
    sim_matrix = cosine_similarity(item_matrix)
    ITEM_SIM_DF = pd.DataFrame(sim_matrix, index=df_encoded.columns, columns=df_encoded.columns)
    
    print("Transactions encoded and similarity matrix calculated.")
    return df_encoded

# 🔹 5. train_models(df_encoded)
def train_models(df_encoded, min_support=0.01):
    print(f"\nTraining FP-Growth with min_support={min_support}...")
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
    return frequent_itemsets

# 🔹 6. generate_rules(frequent_itemsets)
def generate_rules(frequent_itemsets, min_confidence=0.2):
    print(f"Generating rules with min_confidence={min_confidence}...")
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    # Advanced Scoring: (confidence * lift) / log(1 + freq)
    def calculate_advanced_score(row):
        # Get consequence item name (assuming single item consequents for simplicity in scoring)
        consequent = list(row['consequents'])[0]
        freq = ITEM_FREQS.get(consequent, 1)
        return (row['confidence'] * row['lift']) / np.log1p(freq)

    # Filter: confidence > 0.3 and lift > 1.2 (Strict Filtering)
    strong_rules = rules[(rules['confidence'] >= 0.3) & (rules['lift'] >= 1.2)].copy()
    
    if not strong_rules.empty:
        strong_rules['score'] = strong_rules.apply(calculate_advanced_score, axis=1)
    
    print(f"Generated {len(strong_rules)} strong rules.")
    return strong_rules

# 🔹 Matching Logic
def get_best_match(item, all_items):
    item = item.lower().strip()
    if item in PRODUCT_SYNONYMS:
        return PRODUCT_SYNONYMS[item]
    matches = difflib.get_close_matches(item, list(all_items), n=1, cutoff=0.75)
    return matches[0] if matches else None

# 🔹 Hybrid Recommendation Engine
def hybrid_recommend(input_str, rules, k=3):
    """
    Hybrid System: Multi-Input + Association Rules + Similarity Fallback + Category Filter
    """
    input_items = [i.strip() for i in input_str.split('+')]
    all_dataset_items = list(ITEM_FREQS.index)
    
    matched_items = []
    for item in input_items:
        match = get_best_match(item, all_dataset_items)
        if match: matched_items.append(match)
    
    if not matched_items:
        return []

    # Get input categories
    input_cats = set()
    for item in matched_items:
        cat = PRODUCT_CATEGORIES.get(item)
        if cat: input_cats.add(cat)
    
    allowed_cats = set(input_cats)
    for cat in input_cats:
        allowed_cats.update(RELATED_CATEGORIES.get(cat, []))

    recommendations = []
    seen = set(matched_items)

    # 1. Association Rules
    # Look for rules matching as many input items as possible
    matching_rules = rules[rules['antecedents'].apply(lambda x: any(i in x for i in matched_items))].copy()
    
    if not matching_rules.empty:
        # Prioritize rules that match MORE input items
        matching_rules['match_count'] = matching_rules['antecedents'].apply(lambda x: sum(1 for i in matched_items if i in x))
        matching_rules = matching_rules.sort_values(['match_count', 'score'], ascending=False)
        
        for _, row in matching_rules.iterrows():
            for prod in row['consequents']:
                if prod not in seen:
                    prod_cat = PRODUCT_CATEGORIES.get(prod)
                    if not allowed_cats or prod_cat in allowed_cats:
                        recommendations.append({
                            "product": prod,
                            "score": row['score'],
                            "reason": f"{int(row['confidence']*100)}% of users who bought {', '.join(row['antecedents'])} also bought {prod}"
                        })
                        seen.add(prod)
                if len(recommendations) >= k: break
            if len(recommendations) >= k: break

    # 2. Similarity Fallback (if not enough results)
    if len(recommendations) < k:
        sim_results = []
        for item in matched_items:
            if item in ITEM_SIM_DF:
                sim_scores = ITEM_SIM_DF[item].sort_values(ascending=False)
                for prod, score in sim_scores.items():
                    if prod not in seen and score > 0.1: # Cutoff for similarity
                        prod_cat = PRODUCT_CATEGORIES.get(prod)
                        if not allowed_cats or prod_cat in allowed_cats:
                            sim_results.append({
                                "product": prod,
                                "score": score,
                                "reason": "Recommended because it is often similar to your choice"
                            })
        
        # Sort similarity results and add
        sim_results = sorted(sim_results, key=lambda x: x['score'], reverse=True)
        for res in sim_results:
            if len(recommendations) >= k: break
            recommendations.append(res)
            seen.add(res['product'])

    # Normalize scores
    if recommendations:
        # Sort by score descending one last time before normalization
        recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)
        max_score = recommendations[0]['score']
        for rec in recommendations:
            rec['confidence_display'] = round((rec['score'] / max_score) * 100)
            # Ensure no score exceeds 100% (just in case)
            if rec['confidence_display'] > 100: rec['confidence_display'] = 100
    
    return recommendations[:k]

def main():
    dataset_path = 'simple_groceries.csv'
    try:
        raw_df = load_data(dataset_path)
        transactions = create_transactions(raw_df)
        df_encoded = encode_transactions(transactions)
        frequent_itemsets = train_models(df_encoded, min_support=0.01)
        rules = generate_rules(frequent_itemsets, min_confidence=0.2)
        
        print("\n" + "*"*30)
        print("HYBRID INTELLIGENT SYSTEM")
        print("*"*30)
        print("(Type 'exit' to quit, use '+' for multiple items)")
        
        while True:
            user_input = input("\nEnter product(s) (e.g., milk + bread): ").strip()
            if user_input.lower() == 'exit': break
            if not user_input: continue
                
            recs = hybrid_recommend(user_input, rules, k=3)
            print(f"Recommendations for '{user_input}':")
            if recs:
                for i, res in enumerate(recs, 1):
                    print(f" {i}. {res['product']} → {res['confidence_display']}%")
                    print(f"    Reason: {res['reason']}")
            else:
                print(" No specific recommendations found.")
                
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")

if __name__ == "__main__":
    main()
