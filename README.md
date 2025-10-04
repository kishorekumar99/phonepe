
# 📊 PhonePe Pulse Data Analysis & Visualization Project

## 📌 Project Overview
This project is based on the **PhonePe Pulse Dataset**, which contains rich transaction data across India. 
The main goal of this project is to **analyze, process, and visualize** PhonePe transactions to extract meaningful insights.

The project converts raw JSON data into structured formats, stores it in a database, and then visualizes it using Python libraries and web dashboards. 
It provides **year-wise, state-wise, and category-wise** insights into digital payments in India.

---

## 🚩 Problem Statement
Digital transactions in India are growing rapidly, with **PhonePe** being one of the leading UPI-based platforms. 
However, the official PhonePe Pulse dataset is available only as **raw JSON files** which are:
- Large in size
- Nested in structure
- Difficult for direct analysis

**Problem:**  
Stakeholders such as policymakers, businesses, and researchers need **actionable insights** rather than raw data.  
Thus, the challenge is to process and visualize the dataset in a **user-friendly and meaningful way**.

---

## 🎯 Objectives
- Extract raw JSON data from PhonePe Pulse.
- Transform the nested data into structured tables.
- Store the processed data in a **MySQL database** for efficient querying.
- Create **dashboards and visualizations** to highlight:
  - Year-wise transaction trends  
  - State-wise adoption patterns  
  - Category-wise distribution (P2P, Merchant, Recharge, etc.)  
- Provide insights into India’s digital payment growth.

---

## 🏗️ Project Architecture
**Workflow:**  
`Raw JSON Data → Cleaning & Transformation → Database Storage → Visualization → Dashboard`  

**Technologies Used:**
- **Python** (Data processing and backend)
- **Pandas, Plotly, Matplotlib** (Analysis & Visualization)
- **MySQL** (Database storage)
- **Flask / Streamlit** (`app.py` for serving dashboards)
- **GeoJSON + Plotly** (Interactive maps for state-wise analysis)

---

## ⚙️ Methodology
1. **Data Extraction**  
   - Parsed JSON files from PhonePe Pulse dataset.  

2. **Data Transformation**  
   - Cleaned and flattened JSON into structured tables.  
   - Columns: *State, Year, Quarter, Transaction Type, Count, Amount*.  

3. **Database Storage**  
   - Stored data in **MySQL** for fast queries.  

4. **Visualization**  
   - Generated charts and graphs using **Matplotlib & Plotly**.  
   - Built maps with **GeoJSON** for state-level representation.  

5. **Dashboard Deployment**  
   - Created an interactive dashboard using Flask/Streamlit (`app.py`).  

---

## 📈 Features
- Year-wise analysis of PhonePe transactions.  
- State-wise digital payment adoption trends.  
- Category-wise insights (P2P, Merchant, Bill Payments, Recharges).  
- Interactive map visualization of transactions across India.  
- Query support for custom insights.  

---

## 📊 Results
- Clear visualization of digital payment adoption across states.  
- Maharashtra, Karnataka, and Tamil Nadu lead in transaction values.  
- Northeastern and rural states show rapid growth in adoption.  
- P2P transfers and Merchant payments dominate usage trends.  

---

## 🛠️ Challenges
- Handling and flattening large nested JSON data.  
- Mapping state names correctly with GeoJSON files.  
- Optimizing database for large transaction datasets.  

---

## 🚀 Future Scope
- Real-time integration with APIs for live dashboards.  
- Machine learning for transaction trend prediction.  
- Expansion to include **Paytm, Google Pay, BharatPe** datasets.  
- Multi-language dashboard for broader accessibility.  

---

## 📂 Project Structure
```
PhonePe/
│── app.py                  # Flask/Streamlit app for dashboard
│── phonepe_analysis.ipynb  # Jupyter notebook for data analysis
│── database/               # Database scripts/configurations
│── csv_out/                  # Raw PhonePe Pulse dataset (JSON)
│── states_centroids.json   # GeoJSON for map visualization
│── README.md               # Project Documentation
```

---

## ▶️ How to Run the Project
1. Clone this repository:
   ```bash
   git clone <repo-link>
   cd PhonePe
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Setup database (MySQL) and import schema from `/database`.  
4. Run the app:
   ```bash
   python app.py
   ```
5. Open the local server link to explore dashboards.

---

## 🙌 Acknowledgements
- **PhonePe Pulse Dataset** (Official data source)
- Open-source libraries: Pandas, Plotly, Flask, MySQL

---

## ✍️ Author
**[ Kishore kumar]** 
Project: *PhonePe Pulse Data Analysis & Visualization*  
Domain: *FinTech, Data Engineering, Data Visualization*  
