import os
import pandas as pd
import numpy as np

DATA_DIR = "./data"
GLOBAL_SNAPSHOT_CEILING = pd.to_datetime("2025-09-30")
customers = pd.read_csv(f"{DATA_DIR}/customers.csv")
orders = pd.read_csv(f"{DATA_DIR}/orders.csv")
tickets = pd.read_csv(f"{DATA_DIR}/support_tickets.csv")
web_events = pd.read_csv(f"{DATA_DIR}/web_events_snapshot.csv")
labels = pd.read_csv(f"{DATA_DIR}/churn_labels.csv")

# Convert dates and clean up Part 1 anomalies
orders['order_date'] = pd.to_datetime(orders['order_date'])
orders_cleaned = orders[orders['order_date'] <= GLOBAL_SNAPSHOT_CEILING].copy()
orders_cleaned = orders_cleaned[~orders_cleaned['order_id'].astype(str).str.endswith('_DUP')]
gross_99_cap = orders_cleaned['gross_amount'].quantile(0.99)
orders_cleaned['gross_amount'] = np.where(orders_cleaned['gross_amount'] > gross_99_cap, gross_99_cap, orders_cleaned['gross_amount'])

# MULTI-SIGNAL FEATURE ENGINEERING LAYER
print("Engineering RFM features")

# Core RFM Aggregation
rfm_base = orders_cleaned.groupby('customer_id').agg(
    Latest_Purchase_Date=('order_date', 'max'),
    Frequency=('order_id', 'count'),
    Monetary_Value=('gross_amount', 'sum')
).reset_index()

rfm_base['Recency'] = (GLOBAL_SNAPSHOT_CEILING - rfm_base['Latest_Purchase_Date']).dt.days

# Signal 1: Support Complaints Count
ticket_counts = tickets.groupby('customer_id').size().to_frame('Support_Complaints').reset_index()

# Signal 2: Digital Web Activity Interaction Density
web_activity = web_events.groupby('customer_id').size().to_frame('Web_Interactions').reset_index()

# Merge all layers into a single Master Customer Profile Matrix
customer_profile = rfm_base \
    .merge(ticket_counts, on='customer_id', how='left') \
    .merge(web_activity, on='customer_id', how='left') \
    .merge(labels[['customer_id', 'churn_next_60d']], on='customer_id', how='inner')

# Fill missing values for customers with no logs
customer_profile['Support_Complaints'] = customer_profile['Support_Complaints'].fillna(0)
customer_profile['Web_Interactions'] = customer_profile['Web_Interactions'].fillna(0)

# 5-TIER DETERMINISTIC SEGMENTATION MATRIX
print("Categorizing accounts into 5 custom behavioral tiers...")

def assign_five_segments(row):
    # Tier 1: Champions (Recent, frequent, high spend)
    if row['Recency'] <= 30 and row['Frequency'] >= 5:
        return "Champions"
    
    # Tier 2: Loyal Customers (Active frequency, solid monetary, low complaint footprints)
    elif row['Recency'] <= 45 and row['Frequency'] >= 3 and row['Support_Complaints'] <= 1:
        return "Loyal Customers"
    
    # Tier 3: At-Risk Spenders (High historical value but silent too long OR drowning in support friction)
    elif (row['Recency'] > 45 and row['Monetary_Value'] >= 4000) or (row['Support_Complaints'] >= 3 and row['Recency'] <= 60):
        return "At-Risk Spenders"
    
    # Tier 4: Discount-Sensitive (High web page activity/browsing but low order transactions)
    elif row['Web_Interactions'] >= 15 and row['Frequency'] <= 2:
        return "Discount-Sensitive"
    
    # Tier 5: Dormant Customers (Long absence, low interaction, high risk of abandonment)
    else:
        return "Dormant Customers"

customer_profile['Segment'] = customer_profile.apply(assign_five_segments, axis=1)

# Export clean customer matrix mapping
customer_profile[['customer_id', 'Recency', 'Frequency', 'Monetary_Value', 'Support_Complaints', 'Web_Interactions', 'Segment']].to_csv("./segments.csv", index=False)
print("Created segments.csv")

# BUDGET ALLOCATION & STRATEGY MATRICES
print("Allocating constrained campaign budget framework")

segment_metrics = customer_profile.groupby('Segment').agg(
    Customer_Count=('customer_id', 'count'),
    Avg_Recency=('Recency', 'mean'),
    Avg_Frequency=('Frequency', 'mean'),
    Avg_Spend=('Monetary_Value', 'mean'),
    Avg_Complaints=('Support_Complaints', 'mean'),
    Avg_Web=('Web_Interactions', 'mean'),
    Observed_Churn=('churn_next_60d', 'mean')
).reset_index()

segment_metrics['Observed_Churn_Rate (%)'] = (segment_metrics['Observed_Churn'].mul(100)).round(2).astype(str) + '%'

# Defined business strategies and per-capita investment caps
strategy_rules = {
    "Champions": {"Action": "VIP Pre-Access Tokens & Beta Features (No margin erosion)", "Cost": 0.0},
    "Loyal Customers": {"Action": "Milestone Gift Card Reward (₹150 Value)", "Cost": 150.0},
    "At-Risk Spenders": {"Action": "High-Impact Compensation Voucher & Phone Outreach (₹500 Value)", "Cost": 500.0},
    "Discount-Sensitive": {"Action": "Flash Clearance / High-Margin Price-Drop Alert Coupon (₹100 Value)", "Cost": 100.0},
    "Dormant Customers": {"Action": "Low-Cost Automated Email Reactivation Offer (₹20 Value)", "Cost": 20.0}
}

segment_metrics['Retention Action'] = segment_metrics['Segment'].map(lambda x: strategy_rules[x]['Action'])
segment_metrics['Cost Per Cap'] = segment_metrics['Segment'].map(lambda x: strategy_rules[x]['Cost'])
segment_metrics['Segment Allocation'] = segment_metrics['Customer_Count'] * segment_metrics['Cost Per Cap']

# Generate Strategy Markdown Report
retention_matrix_md = f"""# Part 2: Strategic Retention Budget Allocation Matrix

This framework presents our rule-based customer segment classification, merging core RFM vectors with support desk data and web interaction densities.

---

## Strategic Allocation Framework Table

| RFM Segment Name | Volume | Avg Recency | Avg Freq | Avg Spend | Avg Tickets | Avg Web Activity | Segment Churn | Planned Target Retention Action | Cost/Capita | Total Segment Allocation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: |
"""

for _, r in segment_metrics.iterrows():
    retention_matrix_md += f"| **{r['Segment']}** | {int(r['Customer_Count']):,} | {r['Avg_Recency']:.1f}d | {r['Avg_Frequency']:.2f} | ₹{r['Avg_Spend']:,.2f} | {r['Avg_Complaints']:.1f} | {r['Avg_Web']:.1f} | {r['Observed_Churn_Rate (%)']} | {r['Retention Action']} | ₹{r['Cost Per Cap']:.2f} | **₹{r['Segment Allocation']:,.2f}** |\n"

total_prog_cost = segment_metrics['Segment Allocation'].sum()
retention_matrix_md += f"""
---

### Macro Budget Allocation Constraints & Priority Justification
* **Total Global Program Cost Requirements:** **₹{total_prog_cost:,.2f}**
* **CRITICAL CAMPAIGN PRIORITIZATION RUN:** Because our budget envelope is capped, marketing capital must be directed exclusively toward the **At-Risk Spenders** tier first. 
  * *Justification:* This group yields the most immediate ROI protection. They represent large historical capital inflows (Avg Spend: ₹{segment_metrics[segment_metrics['Segment']=='At-Risk Spenders']['Avg_Spend'].values[0]:,.2f}) who are actively detaching due to operational friction (Avg Support Tickets: {segment_metrics[segment_metrics['Segment']=='At-Risk Spenders']['Avg_Complaints'].values[0]:.1f}). Preventing high-value attrition saves far more revenue than trying to reactivate completely cold, low-value dormant users.
"""

with open("./retention_strategy.md", 'w', encoding='utf-8') as f:
    f.write(retention_matrix_md)
print("Created retention_strategy.md")

# NUANCED MANUAL SAFETY AUDITS LAYER (10 EDGES EXTRACOR)
print("10 nuanced customer edge cases")

# Pull 10 profiles from conflicting thresholds (e.g., highly active but massive complaints)
edge_cases = customer_profile[
    ((customer_profile['Support_Complaints'] >= 4) & (customer_profile['Frequency'] >= 4)) | 
    ((customer_profile['Recency'] > 75) & (customer_profile['Monetary_Value'] > 6000))
].head(10)

manual_review_md = """# Manual Review Escalation Cases

This audit highlights **10 specific customer profiles** where behavioral signals conflict, making automated campaign triggers risky.

---

## 🕵️ Customer Edge Case Audit Profiles Matrix

| Selected Customer ID | Recency | Frequency | Gross Spend | Support Tickets | Web Activity | Assigned Segment | Churn Status | Investigative Action & Justification |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :--- |
"""

for _, r in edge_cases.iterrows():
    # Build a truly custom, non-obvious justification based on conflicting metrics
    if r['Support_Complaints'] >= 4:
        justification = "CONFLICT: High-value repeat buyer but experiencing systematic helpdesk failure. **HOLD AUTOMATED OFFERS**. Escalate to Customer Support Lead for high-touch service recovery call within 24 hours."
    else:
        justification = "CONFLICT: Inactive for months but holds massive lifetime monetary weight. A generic promo email looks robotic. **ROUTE TO EXECUTIVE CAMPAIGN**. Dispatch a personalized win-back letter offering dedicated account configuration support."
        
    status = "1 - CHURNED ⚠️" if r['churn_next_60d'] == 1 else "0 - Retained"
    manual_review_md += f"| `{r['customer_id']}` | {r['Recency']}d | {r['Frequency']} | ₹{r['Monetary_Value']:,.2f} | {r['Support_Complaints']} | {r['Web_Interactions']} | {r['Segment']} | {status} | {justification} |\n"

with open("./manual_review_cases.md", 'w', encoding='utf-8') as f:
    f.write(manual_review_md)
print("Created manual_review_cases.md")