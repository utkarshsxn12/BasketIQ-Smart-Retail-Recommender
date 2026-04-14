# 🛒 BasketIQ

An intelligent product recommendation system that analyzes customer purchase patterns using **Market Basket Analysis** to suggest relevant products in real-time.

---

## 🚀 Overview

This project leverages **association rule mining (Apriori & FP-Growth)** to identify frequently bought product combinations and generate smart recommendations.

It also includes a **modern interactive web UI** for real-time user input and dynamic product suggestions.

---

## 🧠 Key Features

* 🔍 Real-time product recommendation
* 📊 Market Basket Analysis (Apriori & FP-Growth)
* 🤖 Hybrid recommendation logic (association + similarity)
* 💡 Explainable AI (shows why a product is recommended)
* 📈 Metrics (confidence, lift, total rules)
* 🎨 Responsive UI (HTML, CSS, JavaScript, Tailwind)

---

## 🛠️ Tech Stack


* Python
* Pandas, NumPy
* mlxtend (Apriori, FP-Growth, Association Rules)
* Scikit-learn (Cosine Similarity)


* HTML, CSS, JavaScript
* Tailwind CSS

---

## ⚙️ How It Works

1. Transaction data is collected and preprocessed
2. Frequent itemsets are generated using Apriori / FP-Growth
3. Association rules are created using confidence & lift
4. Recommendations are ranked and returned
5. UI dynamically displays results with explanation

---

## 📊 Sample Output

Input: `milk`

Output:

* Bread (100%)
* Butter (78%)
* Cheese (65%)

💡 Explanation:
"82% of users who bought milk also bought bread"

---

## 📁 Project Structure

```
├── app.py
├── recommendation_model.py
├── expand_dataset.py
├── Groceries_dataset.csv
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── styles.css
```

---

## ▶️ Run the Project

```bash
# Activate environment
source venv/bin/activate

# Run backend
python app.py
```

Open browser:

```
http://localhost:8000
```

---

## 📈 Model Metrics

* Total Rules: ~1200+
* Avg Confidence: ~47%
* Avg Lift: ~10

---

## 🎯 Future Improvements

* Deep learning-based recommendation
* User personalization
* Deployment (AWS / Vercel)
* Real-time data integration

---

## 👨‍💻 Author

Developed by **Utkarsh Saxena**

---
