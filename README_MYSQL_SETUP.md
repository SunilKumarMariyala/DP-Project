# MySQL Setup Guide for Solar Fault Detection System

This guide provides step-by-step instructions for setting up MySQL for the Solar Fault Detection System. The system requires MySQL to store time-series data from solar panels and support fault detection analytics.

## Prerequisites

- MySQL Server 5.7 or higher
- Python 3.8 or higher with pip
- Administrator access to your computer

## Step 1: Install MySQL Server

### Windows

1. Download the MySQL Installer from [MySQL Downloads](https://dev.mysql.com/downloads/installer/)
2. Run the installer and select "Custom" installation
3. Select the following components:
   - MySQL Server
   - MySQL Workbench
   - Connector/Python
4. Click "Next" and follow the installation wizard
5. Set a root password when prompted (remember this password!)
6. Complete the installation

### macOS

1. Download MySQL from [MySQL Downloads](https://dev.mysql.com/downloads/mysql/)
2. Run the installer package
3. Follow the installation wizard
4. Note the temporary root password provided at the end of installation

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install mysql-server
sudo mysql_secure_installation
```

## Step 2: Verify MySQL Installation

1. Open a terminal or command prompt
2. Connect to MySQL:

```bash
# Windows
mysql -u root -p

# macOS/Linux
sudo mysql -u root -p
```

3. Enter your root password when prompted
4. If you see the MySQL prompt (`mysql>`), the installation was successful
5. Type `exit` to quit the MySQL prompt

## Step 3: Create Database and User

1. Connect to MySQL as root:

```bash
mysql -u root -p
```

2. Create a new database:

```sql
CREATE DATABASE solar_panel_db;
```

3. Create a new user with a secure password:

```sql
CREATE USER 'solar_user'@'localhost' IDENTIFIED BY 'your_secure_password';
```

4. Grant privileges to the user:

```sql
GRANT ALL PRIVILEGES ON solar_panel_db.* TO 'solar_user'@'localhost';
FLUSH PRIVILEGES;
```

5. Exit MySQL:

```sql
EXIT;
```

## Step 4: Install Python MySQL Connector

Install the required Python packages:

```bash
pip install pymysql sqlalchemy
```

## Step 5: Configure Environment Variables

Set the following environment variables with your database connection information:

### Windows PowerShell

```powershell
$env:DB_HOST = "localhost"
$env:DB_USER = "solar_user"
$env:DB_PASSWORD = "your_secure_password"
$env:DB_NAME = "solar_panel_db"
```

### Windows Command Prompt

```cmd
set DB_HOST=localhost
set DB_USER=solar_user
set DB_PASSWORD=your_secure_password
set DB_NAME=solar_panel_db
```

### macOS/Linux

```bash
export DB_HOST=localhost
export DB_USER=solar_user
export DB_PASSWORD=your_secure_password
export DB_NAME=solar_panel_db
```

## Step 6: Initialize Database Schema

Run the database setup script to create the necessary tables:

```bash
python database_setup.py
```

This will create the following tables:
- `solar_panel_data`: Stores solar panel measurements and predictions
- `alerts`: Stores fault alerts
- `system_status`: Stores system status information

## Step 7: Load Sample Data (Optional)

To load sample data for testing:

```bash
python database_setup.py --load-sample-data
```

## Step 8: Verify Database Setup

1. Connect to MySQL:

```bash
mysql -u solar_user -p
```

2. Enter your password when prompted

3. Select the database:

```sql
USE solar_panel_db;
```

4. Verify tables were created:

```sql
SHOW TABLES;
```

5. Check sample data (if loaded):

```sql
SELECT * FROM solar_panel_data LIMIT 10;
```

## Common Issues and Solutions

### Connection Refused

If you get a "Connection refused" error:

1. Check if MySQL is running:
   ```bash
   # Windows
   sc query mysql

   # macOS
   ps aux | grep mysql

   # Linux
   sudo systemctl status mysql
   ```

2. Start MySQL if it's not running:
   ```bash
   # Windows
   net start mysql

   # macOS
   sudo /usr/local/mysql/support-files/mysql.server start

   # Linux
   sudo systemctl start mysql
   ```

### Authentication Error

If you get an authentication error:

1. Reset the user password:
   ```sql
   ALTER USER 'solar_user'@'localhost' IDENTIFIED BY 'new_password';
   FLUSH PRIVILEGES;
   ```

2. Update your environment variables with the new password

### Database Not Found

If the database is not found:

1. Create the database:
   ```sql
   CREATE DATABASE solar_panel_db;
   ```

2. Grant privileges:
   ```sql
   GRANT ALL PRIVILEGES ON solar_panel_db.* TO 'solar_user'@'localhost';
   FLUSH PRIVILEGES;
   ```

## Database Schema

The Solar Fault Detection System uses the following schema:

### solar_panel_data

| Column      | Type      | Description                           |
|-------------|-----------|---------------------------------------|
| id          | INT       | Primary key                           |
| timestamp   | DATETIME  | Time of measurement                   |
| pv_current  | FLOAT     | Solar panel current (A)               |
| pv_voltage  | FLOAT     | Solar panel voltage (V)               |
| power       | FLOAT     | Power output (W)                      |
| temperature | FLOAT     | Cell temperature (°C)                 |
| irradiance  | FLOAT     | Solar irradiance (W/m²)               |
| prediction  | INT       | Fault prediction (0-4)                |
| confidence  | FLOAT     | Prediction confidence (0-1)           |

### alerts

| Column      | Type      | Description                           |
|-------------|-----------|---------------------------------------|
| id          | INT       | Primary key                           |
| timestamp   | DATETIME  | Time of alert                         |
| fault_type  | INT       | Type of fault detected (0-4)          |
| message     | TEXT      | Alert message                         |
| acknowledged| BOOLEAN   | Whether alert was acknowledged        |

### system_status

| Column           | Type      | Description                      |
|------------------|-----------|----------------------------------|
| id               | INT       | Primary key                      |
| timestamp        | DATETIME  | Time of status update            |
| monitoring_active| BOOLEAN   | Whether monitoring is active     |
| last_prediction  | DATETIME  | Time of last prediction          |
| total_predictions| INT       | Total number of predictions made |

## Useful MySQL Commands

### View Recent Data

```sql
SELECT * FROM solar_panel_data ORDER BY timestamp DESC LIMIT 20;
```

### View Fault Distribution

```sql
SELECT prediction, COUNT(*) as count 
FROM solar_panel_data 
GROUP BY prediction 
ORDER BY prediction;
```

### View Unacknowledged Alerts

```sql
SELECT * FROM alerts WHERE acknowledged = 0 ORDER BY timestamp DESC;
```

### Delete Old Data

```sql
DELETE FROM solar_panel_data WHERE timestamp < DATE_SUB(NOW(), INTERVAL 30 DAY);
```

### Backup Database

```bash
mysqldump -u solar_user -p solar_panel_db > backup.sql
```

### Restore Database

```bash
mysql -u solar_user -p solar_panel_db < backup.sql
```

## Next Steps

After setting up MySQL, you can:

1. Run the main application:
   ```bash
   python app.py
   ```

2. Start real-time monitoring:
   ```bash
   python solar_fault_detection.py
   ```

3. Set up continuous data flow from MATLAB:
   ```bash
   python matlab_continuous_demo.py
   ```

For more information, refer to the main README.md file.
