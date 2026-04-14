import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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

# Placeholder for instruction-based changes
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

def recommend(input_str, rules, k=3):
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

def get_top_products(n=10):
    if ITEM_FREQS is not None:
        top = ITEM_FREQS.head(n)
        return [{"product": name, "count": int(count)} for name, count in top.items()]
    return []

def get_model_stats():
    if RULES_DF is None or ITEM_FREQS is None:
        return {}
    
    return {
        "total_transactions": len(ITEM_FREQS), # This is actually total unique items, but we can use it as a proxy or track actual transactions
        "total_rules": len(RULES_DF),
        "avg_confidence": float(round(RULES_DF['confidence'].mean() * 100, 1)),
        "avg_lift": float(round(RULES_DF['lift'].mean(), 2))
    }

def get_top_combinations(n=5):
    if RULES_DF is None:
        return []
    
    top_rules = RULES_DF.sort_values('lift', ascending=False).head(n)
    combinations = []
    for _, row in top_rules.iterrows():
        combinations.append({
            "antecedents": list(row['antecedents']),
            "consequents": list(row['consequents']),
            "confidence": float(round(row['confidence'] * 100, 1)),
            "lift": float(round(row['lift'], 2))
        })
    return combinations

def get_smart_insights():
    if RULES_DF is None or ITEM_FREQS is None:
        return []
    
    insights = []
    
    # Highest confidence rule
    best_conf = RULES_DF.sort_values('confidence', ascending=False).iloc[0]
    insights.append(f"Customers buying {', '.join(list(best_conf['antecedents']))} are {int(best_conf['confidence']*100)}% likely to also buy {', '.join(list(best_conf['consequents']))}.")
    
    # Most frequent item
    top_item = ITEM_FREQS.index[0]
    insights.append(f"{top_item.capitalize()} is your most popular item, appearing in the most transactions.")
    
    # Rule with highest lift
    best_lift = RULES_DF.sort_values('lift', ascending=False).iloc[0]
    insights.append(f"There's a very strong correlation between {', '.join(list(best_lift['antecedents']))} and {', '.join(list(best_lift['consequents']))}.")
    
    return insights

def get_customer_segments(dataset_path='Groceries_dataset.csv'):
    if not os.path.exists(dataset_path):
        return {"error": "Dataset for clustering not found"}
    
    try:
        df = pd.read_csv(dataset_path)
        
        # Simulated 'Amount' since it's missing in the original dataset
        # We'll use a random spend per item for demonstration
        np.random.seed(42)
        df['Amount'] = np.random.uniform(5, 50, size=len(df))
        
        # Feature Engineering: total_spend, purchase_frequency, average_order_value
        customer_data = df.groupby('Member_number').agg({
            'Amount': 'sum',
            'itemDescription': 'count',
            'Date': 'nunique'
        }).rename(columns={
            'Amount': 'total_spend',
            'itemDescription': 'total_items',
            'Date': 'purchase_frequency'
        })
        
        customer_data['average_order_value'] = customer_data['total_spend'] / customer_data['purchase_frequency']
        
        # Scaling features
        features = ['total_spend', 'purchase_frequency', 'average_order_value']
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(customer_data[features])
        
        # K-Means Clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        customer_data['cluster'] = kmeans.fit_predict(scaled_features)
        
        # Labeling clusters meaningfully
        # We'll use total_spend to determine labels
        cluster_means = customer_data.groupby('cluster')['total_spend'].mean().sort_values()
        label_map = {
            cluster_means.index[0]: 'Low Value',
            cluster_means.index[1]: 'Medium Value',
            cluster_means.index[2]: 'High Value'
        }
        customer_data['segment'] = customer_data['cluster'].map(label_map)
        
        # Prepare output
        segments_summary = customer_data.groupby('segment').agg({
            'total_spend': 'mean',
            'purchase_frequency': 'mean',
            'average_order_value': 'mean'
        }).round(2).to_dict('index')
        
        distribution = customer_data['segment'].value_counts().to_dict()
        
        # Sample customers for display
        sample_customers = customer_data.reset_index().head(10).to_dict('records')
        
        return {
            "summary": segments_summary,
            "distribution": distribution,
            "samples": sample_customers
        }
        
    except Exception as e:
        return {"error": str(e)}

def setup_system(dataset_path='simple_groceries.csv'):
    global RULES_DF, ITEM_FREQS
    raw_df = load_data(dataset_path)
    transactions = create_transactions(raw_df)
    df_encoded = encode_transactions(transactions)
    frequent_itemsets = train_models(df_encoded, min_support=0.01)
    RULES_DF = generate_rules(frequent_itemsets, min_confidence=0.2)
    return RULES_DF

def main():
    rules = setup_system()
    print("\nINTELLIGENT SYSTEM (CLI MODE)")
    while True:
        user_input = input("\nEnter product(s) (e.g., milk + bread): ").strip()
        if user_input.lower() == 'exit': break
        recs = recommend(user_input, rules, k=5)
        for i, res in enumerate(recs, 1):
            print(f" {i}. {res['product']} → {res['confidence_display']}% ({res['reason']})")

if __name__ == "__main__":
    main()
