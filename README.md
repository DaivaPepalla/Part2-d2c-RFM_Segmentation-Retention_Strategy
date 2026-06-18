# Customer RFM Segmentation & Behavioral Feature Engineering (Part 2)

This directory houses the foundational feature engineering pipeline for our D2C subscription churn mitigation infrastructure. By transforming raw historical customer metrics into behavioral vectors, this module establishes structured customer profiles across Recency, Frequency, and Monetary dimensions, integrated with app-layer interaction signals.

---

## 🛠️ Project Directory Layout

Ensure your local files are arranged identically to this architecture before initializing the pipeline engine:
```text
Part-2/
├── data/
│   ├── customers.csv             # Raw transactional registration logs
│   ├── orders.csv                # Core historical purchasing records
│   ├── support_tickets.csv       # Customer help desk logging history
│   └── web_events_snapshot.csv   # Aggregated digital app/web login sessions
├── requirements.txt              # Standard package requirements file
├── README.md                     # Operation and execution roadmap
├── retention_strategy.md         # Strategy allocation framework
└── rfm_segmentation.py           # Core feature extraction script

Install Pipeline Requirements
Install the core data engineering stack:
```text
pip install --upgrade pip
pip install pandas==2.2.1 numpy==1.26.4
```
Execution & Output Matrix Generation
Execute the processing script directly from your terminal workspace:
```text
python segment_pipeline.py
```
Automated Outputs & Deliverables
Upon successful completion, the script compiles raw relational customer interaction tables into a standardized feature snapshot file:

segments.csv: The master artifact containing engineered metric attributes per unique customer_id. This file is dynamically tracked and pulled upstream by the Part 3 training and Part 4 deployment engines.

 Engineered Feature Schema DefinitionsThe resulting dataset tracks five high-priority operational signals crucial for identifying customer risk profiles:Target Metric ColumnBehavioral Signal ClassOperational DefinitionRecencyTransactional SilenceCount of calendar days elapsed since the customer's last successful transaction. Higher gaps flag immediate dormancy risks.FrequencyPlatform LoyaltyCumulative transaction volume completed across the history of the account. High counts denote loyal behavior.Monetary_ValueLifetime Financial WeightCumulative gross spend (₹) mapped to the unique subscriber, used to separate premium value tiers from low-tier accounts.Support_ComplaintsProduct/Service FrictionCount of post-purchase support tickets submitted. High frequencies indicate severe system friction.Web_InteractionsApplication EngagementTotal login and usage density tracked across web and mobile layouts, serving as a vital digital health signal.

 Downstream Continuity
The engineered segments.csv file produced in this module serves as the direct data foundation for:

Part 3 Training: Building our Stratified Train/Test splits to calculate model classification baselines.

Part 4 Serving: Validating live API Pydantic model request structures at the microservice gateway level.