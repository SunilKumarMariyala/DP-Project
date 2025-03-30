"""
Display simplified prediction results
"""
import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('solar_panel.db')

# Get prediction counts
print("=== PREDICTION COUNTS ===")
counts_query = """
SELECT prediction, COUNT(*) as count 
FROM solar_panel_data 
WHERE processed_at IS NOT NULL 
GROUP BY prediction
"""
counts_df = pd.read_sql_query(counts_query, conn)
print(counts_df)

# Get latest predictions with key information
print("\n=== LATEST 10 PREDICTIONS ===")
latest_query = """
SELECT 
    id, 
    substr(timestamp, 1, 16) as time, 
    round(pv_current, 2) as current, 
    round(pv_voltage, 2) as voltage, 
    prediction, 
    round(confidence, 2) as confidence
FROM 
    solar_panel_data 
WHERE 
    processed_at IS NOT NULL 
ORDER BY 
    id DESC 
LIMIT 10
"""
latest_df = pd.read_sql_query(latest_query, conn)
print(latest_df)

# Get average confidence by prediction type
print("\n=== AVERAGE CONFIDENCE BY PREDICTION TYPE ===")
confidence_query = """
SELECT 
    prediction, 
    round(AVG(confidence), 2) as avg_confidence
FROM 
    solar_panel_data 
WHERE 
    processed_at IS NOT NULL 
GROUP BY 
    prediction
"""
confidence_df = pd.read_sql_query(confidence_query, conn)
print(confidence_df)

# Close connection
conn.close()
