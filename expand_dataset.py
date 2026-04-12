import random

def generate_expanded_dataset(file_path, total_transactions=5000):
    """
    Generates a high-volume, structured grocery dataset with Blinkit-style categories.
    """
    # 1. Define Expanded Blinkit-style Pattern Templates
    patterns = {
        # Fresh Produce
        "fruits": ["apple", "banana", "orange", "grapes", "mango", "watermelon", "kiwi", "papaya", "pomegranate", "guava", "pineapple", "pear"],
        "vegetables": ["tomato", "onion", "potato", "cucumber", "spinach", "carrot", "peas", "capsicum", "broccoli", "cauliflower", "lady finger", "ginger", "garlic", "coriander", "green chili"],
        
        # Staples & Pantry
        "staples": ["rice", "wheat flour", "dal", "toor dal", "moong dal", "sugar", "salt", "cooking oil", "ghee", "mustard oil", "turmeric", "red chili powder", "jeera"],
        "bakery_dairy": ["milk", "bread", "butter", "cheese", "yogurt", "eggs", "paneer", "fresh cream", "brown bread", "sandwich bread"],
        
        # Breakfast & Instant Food
        "breakfast": ["oats", "cornflakes", "honey", "juice", "muesli", "peanut butter", "jam", "eggs", "maggi", "cup noodles"],
        "tea_coffee_time": ["tea", "coffee", "biscuits", "cookies", "cake", "rusk", "green tea", "instant coffee", "sugar"],
        
        # Snacks & Munchies
        "munchies": ["chips", "nachos", "popcorn", "chocolate", "salted peanuts", "bhujia", "kurkure", "mixture", "roasted cashews"],
        "beverages": ["soft drink", "energy drink", "fruit juice", "coconut water", "iced tea", "soda", "mineral water"],
        
        # Personal Care & Household
        "personal_care": ["soap", "shampoo", "toothpaste", "toothbrush", "face wash", "hand wash", "body lotion", "deodorant", "shaving cream"],
        "household_items": ["detergent", "dishwash gel", "toilet cleaner", "floor cleaner", "garbage bags", "toilet paper", "kitchen roll", "napkins"],
        
        # Protein & Gourmet
        "meat_protein": ["eggs", "chicken", "tofu", "paneer", "soya chunks", "mutton", "fish", "frozen peas", "sweet corn"]
    }

    # 2. Pattern keys for selection
    pattern_keys = list(patterns.keys())
    
    transactions = []

    for _ in range(total_transactions):
        # Choose a primary category
        main_cat = random.choice(pattern_keys)
        items_pool = patterns[main_cat]
        
        # Transaction size (2 to 6 items)
        size = random.randint(2, min(6, len(items_pool)))
        
        # Sample items from the pool
        transaction = random.sample(items_pool, size)
        
        # Add logical cross-category pairings (Blinkit style)
        rand_val = random.random()
        if rand_val < 0.3: # 30% chance to add staples to fresh
            if main_cat in ["fruits", "vegetables"]:
                transaction.append(random.choice(patterns["staples"]))
        elif rand_val < 0.5: # 20% chance to add dairy to breakfast
            if main_cat == "breakfast":
                transaction.append(random.choice(patterns["bakery_dairy"]))
        elif rand_val < 0.65: # 15% chance to add munchies to beverages
            if main_cat == "beverages":
                transaction.append(random.choice(patterns["munchies"]))
        
        # Ensure 'milk' and 'bread' are frequently recurring across common categories
        if main_cat in ["bakery_dairy", "breakfast", "tea_coffee_time"] and random.random() < 0.4:
            if "milk" not in transaction: transaction.append("milk")
            if "bread" not in transaction: transaction.append("bread")

        # Sort for consistency and remove duplicates
        transaction = sorted(list(set(transaction)))
        transactions.append(",".join(transaction))

    # 3. Save to CSV
    with open(file_path, 'w') as f:
        for t in transactions:
            f.write(t + "\n")
            
    print(f"Successfully expanded dataset to {total_transactions} Blinkit-style transactions.")

if __name__ == "__main__":
    # Expand to 5000 transactions as requested
    generate_expanded_dataset('simple_groceries.csv', 5000)
