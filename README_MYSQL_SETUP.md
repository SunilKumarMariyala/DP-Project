# MySQL Setup for Solar Fault Detection System

This guide explains how to set up MySQL for the Solar Fault Detection System.

## Prerequisites

- MySQL Server 8.0 or later
- Python 3.7 or later with required packages (see `requirements.txt`)

## Installation Steps

### 1. Install MySQL Server

#### Windows
1. Download MySQL Installer from [MySQL Community Downloads](https://dev.mysql.com/downloads/mysql/)
2. Run the installer and select:
   - MySQL Server
   - MySQL Workbench (optional but recommended)
   - Connector/Python
3. Follow the installation wizard
4. Set a root password when prompted (remember this password!)

### 2. Create Database User (Optional but Recommended)

For better security, create a dedicated user for the application instead of using root:

1. Open MySQL Workbench or MySQL Command Line Client
2. Connect using the root account
3. Execute the following commands:

```sql
CREATE USER 'solar_user'@'localhost' IDENTIFIED BY 'your_secure_password';
GRANT ALL PRIVILEGES ON solar_panel_db.* TO 'solar_user'@'localhost';
FLUSH PRIVILEGES;
```

### 3. Configure the Application

Update your command line arguments when running the application:

```bash
python solar_fault_detection.py --db-user=solar_user --db-password=your_secure_password --db-name=solar_panel_db
```

Or use the default values by editing the global variables in the script:

```python
DB_USER = 'solar_user'  # Replace with your MySQL username
DB_PASSWORD = 'your_secure_password'  # Replace with your MySQL password
DB_NAME = 'solar_panel_db'
```

## Database Schema

The application will automatically create the following tables:

1. `solar_data` - Stores solar panel measurements
2. `predictions` - Stores fault detection predictions
3. `alerts` - Stores alerts generated from predictions

## Troubleshooting

### Connection Issues

If you encounter connection issues:

1. Verify MySQL service is running
2. Check credentials are correct
3. Ensure the database exists
4. Verify network connectivity if using a remote database

### Permission Issues

If you encounter permission issues:

```sql
GRANT ALL PRIVILEGES ON *.* TO 'solar_user'@'localhost';
FLUSH PRIVILEGES;
```

## Backup and Restore

### Backup Database

```bash
mysqldump -u solar_user -p solar_panel_db > solar_panel_backup.sql
```

### Restore Database

```bash
mysql -u solar_user -p solar_panel_db < solar_panel_backup.sql
```
