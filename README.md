# E-Commerce Purchase Prediction

Predict whether website visitors will make a purchase using machine learning.

## Project Structure

```
├── app.py                 # Streamlit web application
├── train_model.py         # Model training script
├── eval_model.py          # Model evaluation script
├── data/                  # Dataset directory
│   └── online_shoppers_intention.csv
├── models/                # Trained model files
│   └── model_prediksi_pembelian_ecommerce.pkl
├── outputs/               # Evaluation results
├── notebooks/             # Jupyter notebooks
│   └── prediksi_pembelian_ecommerce.ipynb
├── requirements.txt       # Python dependencies
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Run Web App
```bash
streamlit run app.py
```

### Train Model
```bash
python train_model.py
```

### Evaluate Model
```bash
python eval_model.py
```

## Model Details

- **Algorithm**: XGBoost with SMOTE for class balancing
- **Features**: 17 visitor behavior metrics
- **Target**: Purchase prediction (0 = No Purchase, 1 = Purchase)
- **Training data**: UCI Online Shoppers Intention dataset

---

## Field Guide — What is each field? Where do I get it?

This tool predicts purchase likelihood using **17 pieces of data** about a visitor's session.
Below is a plain-English explanation of every field, where to find it, and how it's collected.

### 📅 Session Info

**Month**  
*What it is:* The calendar month when the visit happened.  
*Where to get it:* Google Analytics → Audience → Overview → Primary Dimension: Month.  
*Why it matters:* Shopping behavior changes with seasons. November (Black Friday) and December
(Christmas) have much higher purchase rates than February.

**Weekend**  
*What it is:* Whether the visit occurred on Saturday or Sunday.  
*Where to get it:* Google Analytics calculates this automatically from the session timestamp.  
*Why it matters:* Weekend shoppers may have different intent than weekday browsers.

---

### 👤 Visitor Type

**Visitor type**  
*What it is:* Whether this person has visited your website before.  
- **Returning Visitor** — has visited before, possibly made a purchase before.  
- **New Visitor** — first time on your website.  
- **Other** — rare category (e.g. bot, unclassified).  

*Where to get it:* Google Analytics → Audience → Behavior → New vs Returning.  
*Why it matters:* Returning visitors are more likely to buy because they already trust the brand.
About 77% of purchasers in the training data were returning visitors.

---

### 📄 Page Visits

**Product pages visited**  
*What it is:* How many product listing or product detail pages the visitor viewed.  
*Where to get it:* Google Analytics → Behavior → Site Content → All Pages. Filter by pages
containing your product URLs (e.g. `/product/`, `/p/`). Count sessions that visited these pages.  
*Why it matters:* More product page views = higher purchase intent. Purchasers in the training
data visited an average of **48 product pages**, while non-purchasers visited only **29**.

**Time on product pages (minutes)**  
*What it is:* Total time the visitor spent reading product pages.  
*Where to get it:* Google Analytics → Behavior → Site Content → All Pages → Average Time on Page.
Multiply by number of product pages visited.  
*Why it matters:* Longer time on product pages suggests the visitor is researching,
comparing, or reading reviews — strong purchase signals.

**Admin pages visited**  
*What it is:* Pages like login, account settings, order tracking, help center.  
*Where to get it:* Google Analytics → All Pages → filter by `/login`, `/account`, `/help`, etc.  
*Why it matters:* Visitors managing accounts are engaged but not necessarily shopping.
Low relevance to purchase prediction.

**Info pages visited**  
*What it is:* Pages like FAQ, about us, shipping info, return policy.  
*Where to get it:* Google Analytics → All Pages → filter by `/faq`, `/about`, `/shipping`, etc.  
*Why it matters:* Visitors reading shipping/return policies are close to buying — they want
to confirm details before purchasing.

---

### 📈 Behavior Metrics

**Bounce rate**  
*What it is:* Percentage of sessions where the visitor left after viewing only **one page**,
without clicking anything else.  
*Where to get it:* Google Analytics → Audience → Overview → Bounce Rate column.  
*Why it matters:* High bounce rate = visitor wasn't interested. Purchasers almost never bounce
(average bounce rate: 0.5%). Non-purchasers average 2.3%.

**Exit rate**  
*What it is:* Percentage of visitors who left the website from a specific page.  
*Where to get it:* Google Analytics → Behavior → Site Content → Exit Pages.  
*Why it matters:* High exit rate on product pages = visitors leaving without buying.
Low exit rate = visitors continue browsing or convert.

**Page Value** ⭐ *(most important)*  
*What it is:* The average monetary value of a page, calculated from all sessions that
included that page **and ended in a purchase**. Pages with Page Value are the ones that
typically lead to sales.  
*Where to get it:* Google Analytics → Behavior → Site Content → All Pages →
look for the **"Page Value"** column.  
- If a page has Page Value = $20, it means visitors who viewed that page
  ended up spending an average of $20 on your site.  
- Pages with $0 Page Value never led to a purchase in the data.  
*Why it matters:* This is the **strongest predictor** in the model. Even a small Page Value
($5+) dramatically increases purchase probability.  
*How it's collected:* Google Analytics tracks every page view, and when a visitor completes
a transaction, it attributes a portion of that transaction value back to all pages the
visitor viewed during that session.

**Special day proximity**  
*What it is:* How close the visit was to a special day (Valentine's, Mother's Day,
Black Friday, Christmas, etc.). 0 = no special day, 1 = exactly on the special day.  
*Where to get it:* This is a calculated field. Mark special shopping days in your calendar
and set this to 0.8–1.0 on those days, 0 otherwise.  
*Why it matters:* Special days drive purchase intent. Black Friday sessions are far more
likely to convert than a random Tuesday in March.

---

### 🔧 Technical Details (Anonymized)

**Operating system, Browser, Region, Traffic type**  
*What it is:* Anonymized IDs representing the visitor's device type, browser, geographic
region, and how they found the website (search, direct, social, etc.).  
*Where to get it:* Google Analytics → Audience → Technology (OS, Browser) or
Geo → Region, or Acquisition → All Traffic → Channels (Traffic type).  
*Why it matters:* These are minor factors. The model uses them as tie-breakers.
**You can safely leave these at their defaults** — they are pre-set to the most common
values and changing them won't significantly affect the prediction.

---

### 💡 Quick Summary

| Field | Where to find it | Impact on prediction |
|---|---|---|
| **Page Value** | GA → All Pages → Page Value | ⭐⭐⭐⭐⭐ Very high |
| **Product pages visited** | GA → All Pages (filter product URLs) | ⭐⭐⭐⭐ High |
| **Time on product pages** | GA → All Pages → Avg Time × count | ⭐⭐⭐ Medium |
| **Bounce rate** | GA → Overview → Bounce Rate | ⭐⭐⭐ Medium |
| **Visitor type** | GA → Audience → New vs Returning | ⭐⭐ Medium |
| **Month** | GA → Overview → Month | ⭐ Low |
| **Exit rate** | GA → Exit Pages | ⭐ Low |
| **Weekend** | GA → Overview → Day of Week | ⭐ Low |
| **OS / Browser / Region / Traffic** | GA → Technology / Geo / Acquisition | ⭐ Very low (use defaults) |