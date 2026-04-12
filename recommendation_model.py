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

# State management
ITEM_SIM_DF = None
ITEM_FREQS = None
RULES_DF = None

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            items = [item.strip() for item in line.split(',') if item.strip()]
            if items: data.append(items)
    df = pd.DataFrame(data)
    return df

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
    return transactions

def encode_transactions(transactions):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    global ITEM_SIM_DF
    item_matrix = df_encoded.T.values
    sim_matrix = cosine_similarity(item_matrix)
    ITEM_SIM_DF = pd.DataFrame(sim_matrix, index=df_encoded.columns, columns=df_encoded.columns)
    return df_encoded

def train_models(df_encoded, min_support=0.01):
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
    return frequent_itemsets

def generate_rules(frequent_itemsets, min_confidence=0.2):
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    def calculate_advanced_score(row):
        consequent = list(row['consequents'])[0]
        freq = ITEM_FREQS.get(consequent, 1)
        return (row['confidence'] * row['lift']) / np.log1p(freq)

    strong_rules = rules[(rules['confidence'] >= 0.3) & (rules['lift'] >= 1.2)].copy()
    if not strong_rules.empty:
        strong_rules['score'] = strong_rules.apply(calculate_advanced_score, axis=1)
    return strong_rules

def get_best_match(item, all_items):
    item = item.lower().strip()
    if item in PRODUCT_SYNONYMS:
        return PRODUCT_SYNONYMS[item]
    matches = difflib.get_close_matches(item, list(all_items), n=1, cutoff=0.75)
    return matches[0] if matches else None

def hybrid_recommend(input_str, rules, k=3):
    input_items = [i.strip() for i in input_str.split('+')]
    all_dataset_items = list(ITEM_FREQS.index)
    
    matched_items = []
    for item in input_items:
        match = get_best_match(item, all_dataset_items)
        if match: matched_items.append(match)
    
    if not matched_items: return []

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
                        # 🔹 Improved Sentence Generation Logic
                        antecedents_list = list(row['antecedents'])
                        if len(antecedents_list) > 1:
                            antecedents_text = ", ".join(antecedents_list[:-1]) + " and " + antecedents_list[-1]
                        else:
                            antecedents_text = antecedents_list[0]
                            
                        confidence_pct = int(row['confidence']*100)
                        reason = f"{confidence_pct}% of users who bought {antecedents_text} also bought {prod}"
                        
                        recommendations.append({
                            "product": prod,
                            "score": row['score'],
                            "reason": reason
                        })
                        seen.add(prod)
                if len(recommendations) >= k: break
            if len(recommendations) >= k: break

    # 2. Similarity Fallback
    if len(recommendations) < k:
        sim_results = []
        # Create a display string for input items for the similarity reason
        if len(matched_items) > 1:
            input_text = ", ".join(matched_items[:-1]) + " and " + matched_items[-1]
        else:
            input_text = matched_items[0]

        for item in matched_items:
            if item in ITEM_SIM_DF:
                sim_scores = ITEM_SIM_DF[item].sort_values(ascending=False)
                for prod, score in sim_scores.items():
                    if prod not in seen and score > 0.1:
                        prod_cat = PRODUCT_CATEGORIES.get(prod)
                        if not allowed_cats or prod_cat in allowed_cats:
                            sim_results.append({
                                "product": prod,
                                "score": score,
                                "reason": f"Recommended because it is often similar to your choice of {input_text}"
                            })
        sim_results = sorted(sim_results, key=lambda x: x['score'], reverse=True)
        for res in sim_results:
            if len(recommendations) >= k: break
            recommendations.append(res)
            seen.add(res['product'])

    if recommendations:
        recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)
        max_score = recommendations[0]['score']
        for rec in recommendations:
            rec['confidence_display'] = round((rec['score'] / max_score) * 100)
            if rec['confidence_display'] > 100: rec['confidence_display'] = 100
    
    return recommendations[:k]

def setup_system(dataset_path='simple_groceries.csv'):
    global RULES_DF
    raw_df = load_data(dataset_path)
    transactions = create_transactions(raw_df)
    df_encoded = encode_transactions(transactions)
    frequent_itemsets = train_models(df_encoded, min_support=0.01)
    RULES_DF = generate_rules(frequent_itemsets, min_confidence=0.2)
    return RULES_DF

def main():
    rules = setup_system()
    print("\nHYBRID INTELLIGENT SYSTEM (CLI MODE)")
    while True:
        user_input = input("\nEnter product(s) (e.g., milk + bread): ").strip()
        if user_input.lower() == 'exit': break
        recs = hybrid_recommend(user_input, rules, k=5)
        for i, res in enumerate(recs, 1):
            print(f" {i}. {res['product']} → {res['confidence_display']}% ({res['reason']})")

if __name__ == "__main__":
    main()
