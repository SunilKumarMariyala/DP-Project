# Comprehensive MySQL Installation and Setup Guide

This guide provides an in-depth, step-by-step approach to installing and configuring MySQL for the Solar Fault Detection System. This guide covers installation on multiple platforms, advanced configuration options, security considerations, and performance tuning.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [MySQL Architecture Overview](#mysql-architecture-overview)
3. [Detailed Installation Process](#detailed-installation-process)
   - [Windows Installation](#windows-installation)
   - [macOS Installation](#macos-installation)
   - [Linux Installation](#linux-installation)
4. [Post-Installation Configuration](#post-installation-configuration)
5. [Security Hardening](#security-hardening)
6. [Performance Optimization](#performance-optimization)
7. [Database and User Setup](#database-and-user-setup)
8. [Configuring for Solar Fault Detection System](#configuring-for-solar-fault-detection-system)
9. [Backup and Recovery](#backup-and-recovery)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Topics](#advanced-topics)

## Prerequisites

Before installing MySQL, ensure your system meets these requirements:

### Hardware Requirements
- **CPU**: 2+ cores recommended (4+ cores for production)
- **RAM**: Minimum 4GB (8GB+ recommended for production)
- **Disk Space**: At least 5GB free space (SSD recommended)
- **Network**: Stable internet connection for installation

### Software Requirements
- **Windows**: Windows 10/11 or Windows Server 2016/2019/2022
- **macOS**: macOS 10.15 (Catalina) or newer
- **Linux**: Ubuntu 20.04+, Debian 10+, RHEL/CentOS 8+, or other modern distributions
- **Python**: Python 3.8+ with pip

## MySQL Architecture Overview

Understanding MySQL's architecture helps with proper configuration:

### Key Components
- **MySQL Server**: Core database engine
- **Storage Engines**: InnoDB (default), MyISAM, Memory, etc.
- **Connection Pool**: Manages client connections
- **Query Cache**: Caches query results (deprecated in MySQL 8.0+)
- **Parser**: Parses SQL statements
- **Optimizer**: Optimizes query execution plans
- **Buffer Pool**: In-memory cache for data and indexes
- **Log Files**: Binary logs, error logs, general logs, slow query logs

### Storage Engines
The Solar Fault Detection System uses InnoDB (the default) for:
- ACID compliance (Atomicity, Consistency, Isolation, Durability)
- Row-level locking for better concurrency
- Foreign key constraints
- Crash recovery

## Detailed Installation Process

### Windows Installation

#### Step 1: Download MySQL Installer
1. Go to the [MySQL Downloads page](https://dev.mysql.com/downloads/installer/)
2. Download the MySQL Installer for Windows (mysql-installer-web-community-x.x.x.x.msi)
3. Verify the download using the provided checksum

#### Step 2: Run the Installer
1. Right-click the installer and select "Run as administrator"
2. Accept the license agreement
3. Choose "Custom" installation type for full control

#### Step 3: Select Components
For a complete installation, select:
- MySQL Server (required)
- MySQL Workbench (GUI tool)
- MySQL Shell (command-line tool)
- Connector/Python (for Python integration)
- MySQL Router (if using replication)
- Sample Databases (optional)

#### Step 4: Installation Configuration
1. **Server Configuration Type**:
   - Development Computer: For development environments
   - Server Computer: For production servers
   - Dedicated Computer: For dedicated database servers

2. **Connectivity**:
   - TCP/IP Port: Default is 3306 (change if needed)
   - X Protocol Port: Default is 33060 (for X Protocol)
   - Open Windows Firewall ports: Recommended

3. **Authentication Method**:
   - Use Strong Password Encryption (recommended)
   - Use Legacy Authentication Method (only if required by legacy applications)

4. **Accounts and Roles**:
   - Set MySQL root password (use a strong, complex password)
   - Add MySQL user accounts if needed

5. **Windows Service**:
   - Configure MySQL as a Windows Service (recommended)
   - Service Name: MySQL80 (default)
   - Start at System Startup: Yes (recommended)
   - Run Windows Service as: Standard System Account (default)

6. **Apply Configuration**:
   - Review settings
   - Click "Execute" to apply configuration
   - Note any errors or warnings

#### Step 5: Verify Installation
1. Open Command Prompt as administrator
2. Verify MySQL service is running:
   ```cmd
   sc query mysql80
   ```
3. Connect to MySQL:
   ```cmd
   "C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe" -u root -p
   ```
4. Enter your root password when prompted
5. If you see the MySQL prompt (`mysql>`), installation was successful

### macOS Installation

#### Step 1: Download MySQL Package
1. Go to [MySQL Downloads](https://dev.mysql.com/downloads/mysql/)
2. Select macOS from the dropdown menu
3. Download the DMG archive for your macOS version
4. Verify the download using the provided checksum

#### Step 2: Install MySQL
1. Open the downloaded DMG file
2. Double-click the MySQL installer package
3. Follow the installation wizard
4. Choose "Use Legacy Password Encryption" only if needed for compatibility
5. Note the temporary root password shown at the end of installation (very important!)

#### Step 3: Configure MySQL as a Service
1. Open System Preferences
2. Click on the MySQL icon
3. Enable "Start MySQL when your computer starts up"
4. Click "Start MySQL Server" if it's not already running

#### Step 4: Secure MySQL Installation
1. Open Terminal
2. Connect to MySQL using the temporary password:
   ```bash
   /usr/local/mysql/bin/mysql -u root -p
   ```
3. Enter the temporary password when prompted
4. Change the root password:
   ```sql
   ALTER USER 'root'@'localhost' IDENTIFIED BY 'your_new_secure_password';
   ```
5. Exit MySQL:
   ```sql
   EXIT;
   ```

#### Step 5: Add MySQL to PATH
1. Open Terminal
2. Edit your shell profile file:
   ```bash
   # For Bash
   echo 'export PATH=$PATH:/usr/local/mysql/bin' >> ~/.bash_profile
   source ~/.bash_profile
   
   # For Zsh
   echo 'export PATH=$PATH:/usr/local/mysql/bin' >> ~/.zshrc
   source ~/.zshrc
   ```
3. Verify MySQL is in your PATH:
   ```bash
   which mysql
   ```

### Linux Installation

#### Ubuntu/Debian Installation

##### Step 1: Update Package Repository
```bash
sudo apt update
sudo apt upgrade -y
```

##### Step 2: Install MySQL Server
```bash
sudo apt install mysql-server -y
```

##### Step 3: Check MySQL Service Status
```bash
sudo systemctl status mysql
```

##### Step 4: Secure MySQL Installation
```bash
sudo mysql_secure_installation
```
Follow the prompts to:
- Set up VALIDATE PASSWORD component (recommended)
- Set a strong root password
- Remove anonymous users (recommended)
- Disallow root login remotely (recommended)
- Remove test database (recommended)
- Reload privilege tables (recommended)

##### Step 5: Configure MySQL to Start on Boot
```bash
sudo systemctl enable mysql
```

#### RHEL/CentOS Installation

##### Step 1: Add MySQL Repository
```bash
sudo rpm -Uvh https://repo.mysql.com/mysql80-community-release-el8-1.noarch.rpm
sudo dnf module disable mysql
```

##### Step 2: Install MySQL Server
```bash
sudo dnf install mysql-community-server -y
```

##### Step 3: Start MySQL Service
```bash
sudo systemctl start mysqld
```

##### Step 4: Get Temporary Root Password
```bash
sudo grep 'temporary password' /var/log/mysqld.log
```

##### Step 5: Secure MySQL Installation
```bash
sudo mysql_secure_installation
```

##### Step 6: Configure MySQL to Start on Boot
```bash
sudo systemctl enable mysqld
```

## Post-Installation Configuration

### MySQL Configuration File
The main configuration file is:
- Windows: `C:\ProgramData\MySQL\MySQL Server 8.0\my.ini`
- macOS: `/usr/local/mysql/etc/my.cnf`
- Linux: `/etc/mysql/my.cnf` or `/etc/my.cnf`

### Essential Configuration Parameters

#### Basic Settings
```ini
[mysqld]
# Server identification
server-id = 1
port = 3306

# Character set and collation
character-set-server = utf8mb4
collation-server = utf8mb4_unicode_ci

# Default storage engine
default-storage-engine = InnoDB

# Error log
log_error = mysql-error.log
```

#### InnoDB Settings
```ini
[mysqld]
# InnoDB buffer pool size (50-80% of RAM)
innodb_buffer_pool_size = 1G

# InnoDB log file size
innodb_log_file_size = 256M

# InnoDB flush method (O_DIRECT for Linux)
innodb_flush_method = O_DIRECT

# InnoDB flush log at transaction commit
innodb_flush_log_at_trx_commit = 1
```

#### Connection Settings
```ini
[mysqld]
# Maximum connections
max_connections = 151

# Connection timeout
connect_timeout = 10

# Wait timeout
wait_timeout = 28800

# Interactive timeout
interactive_timeout = 28800
```

### Applying Configuration Changes
1. Edit the configuration file with your preferred settings
2. Save the file
3. Restart MySQL:
   - Windows: `net stop mysql80 && net start mysql80`
   - macOS: `sudo /usr/local/mysql/support-files/mysql.server restart`
   - Linux: `sudo systemctl restart mysql`

## Security Hardening

### Network Security
1. Bind MySQL to localhost if not accessed remotely:
   ```ini
   [mysqld]
   bind-address = 127.0.0.1
   ```

2. Configure firewall to restrict access:
   - Windows: Use Windows Firewall
   - macOS: Use built-in firewall
   - Linux: Use iptables or ufw
   ```bash
   # Linux (ufw)
   sudo ufw allow from trusted_ip_address to any port 3306
   ```

### User Account Security
1. Use strong passwords for all accounts
2. Create specific users with minimal privileges:
   ```sql
   CREATE USER 'solar_user'@'localhost' IDENTIFIED BY 'strong_password';
   GRANT SELECT, INSERT, UPDATE, DELETE ON solar_panel_db.* TO 'solar_user'@'localhost';
   ```

3. Remove anonymous users:
   ```sql
   DELETE FROM mysql.user WHERE User='';
   ```

4. Disable remote root access:
   ```sql
   DELETE FROM mysql.user WHERE User='root' AND Host NOT IN ('localhost', '127.0.0.1', '::1');
   ```

### Encryption
1. Enable SSL/TLS for encrypted connections:
   ```ini
   [mysqld]
   ssl-ca=/path/to/ca.pem
   ssl-cert=/path/to/server-cert.pem
   ssl-key=/path/to/server-key.pem
   ```

2. Require SSL for specific users:
   ```sql
   ALTER USER 'solar_user'@'localhost' REQUIRE SSL;
   ```

### Audit Logging
Enable audit logging to track database activities:
```ini
[mysqld]
audit_log_file = /var/log/mysql/audit.log
audit_log_format = JSON
audit_log_policy = ALL
```

## Performance Optimization

### Buffer Pool Configuration
The InnoDB buffer pool is the most important setting for performance:
```ini
[mysqld]
# Set to 50-80% of available RAM
innodb_buffer_pool_size = 4G

# Multiple buffer pool instances for high concurrency
innodb_buffer_pool_instances = 4
```

### Query Cache (MySQL 5.7 and earlier)
```ini
[mysqld]
query_cache_type = 1
query_cache_size = 64M
query_cache_limit = 2M
```
Note: Query cache is removed in MySQL 8.0+

### Thread Pool
```ini
[mysqld]
thread_handling = pool-of-threads
thread_pool_size = 16
```

### Temporary Tables
```ini
[mysqld]
tmp_table_size = 64M
max_heap_table_size = 64M
```

### Join Buffer
```ini
[mysqld]
join_buffer_size = 2M
```

### Sort Buffer
```ini
[mysqld]
sort_buffer_size = 4M
```

### Read/Write Buffers
```ini
[mysqld]
read_buffer_size = 2M
read_rnd_buffer_size = 4M
```

## Database and User Setup

### Creating the Database
```sql
CREATE DATABASE solar_panel_db
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;
```

### Creating a Database User
```sql
-- Create user with password
CREATE USER 'solar_user'@'localhost' IDENTIFIED BY 'your_secure_password';

-- Grant privileges
GRANT ALL PRIVILEGES ON solar_panel_db.* TO 'solar_user'@'localhost';

-- Apply privileges
FLUSH PRIVILEGES;
```

### Setting Resource Limits
```sql
-- Limit connections per hour
ALTER USER 'solar_user'@'localhost' WITH 
    MAX_QUERIES_PER_HOUR 10000
    MAX_UPDATES_PER_HOUR 5000
    MAX_CONNECTIONS_PER_HOUR 1000
    MAX_USER_CONNECTIONS 20;
```

### Creating Database Schema
For the Solar Fault Detection System, create the necessary tables:

```sql
USE solar_panel_db;

-- Solar panel data table
CREATE TABLE solar_panel_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    pv_current FLOAT NOT NULL,
    pv_voltage FLOAT NOT NULL,
    power FLOAT NOT NULL,
    temperature FLOAT,
    irradiance FLOAT,
    prediction INT,
    confidence FLOAT,
    INDEX idx_timestamp (timestamp),
    INDEX idx_prediction (prediction)
) ENGINE=InnoDB;

-- Alerts table
CREATE TABLE alerts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    fault_type INT NOT NULL,
    message TEXT,
    acknowledged BOOLEAN DEFAULT FALSE,
    INDEX idx_timestamp (timestamp),
    INDEX idx_acknowledged (acknowledged)
) ENGINE=InnoDB;

-- System status table
CREATE TABLE system_status (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    monitoring_active BOOLEAN DEFAULT FALSE,
    last_prediction DATETIME,
    total_predictions INT DEFAULT 0,
    INDEX idx_timestamp (timestamp)
) ENGINE=InnoDB;
```

## Configuring for Solar Fault Detection System

### Environment Variables
Set up environment variables for the application:

#### Windows PowerShell
```powershell
$env:DB_HOST = "localhost"
$env:DB_USER = "solar_user"
$env:DB_PASSWORD = "your_secure_password"
$env:DB_NAME = "solar_panel_db"

# Make permanent (optional)
[System.Environment]::SetEnvironmentVariable("DB_HOST", "localhost", "User")
[System.Environment]::SetEnvironmentVariable("DB_USER", "solar_user", "User")
[System.Environment]::SetEnvironmentVariable("DB_PASSWORD", "your_secure_password", "User")
[System.Environment]::SetEnvironmentVariable("DB_NAME", "solar_panel_db", "User")
```

#### Windows Command Prompt
```cmd
set DB_HOST=localhost
set DB_USER=solar_user
set DB_PASSWORD=your_secure_password
set DB_NAME=solar_panel_db

:: Make permanent (optional)
setx DB_HOST "localhost"
setx DB_USER "solar_user"
setx DB_PASSWORD "your_secure_password"
setx DB_NAME "solar_panel_db"
```

#### Linux/macOS
```bash
export DB_HOST=localhost
export DB_USER=solar_user
export DB_PASSWORD=your_secure_password
export DB_NAME=solar_panel_db

# Make permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export DB_HOST=localhost' >> ~/.bashrc
echo 'export DB_USER=solar_user' >> ~/.bashrc
echo 'export DB_PASSWORD=your_secure_password' >> ~/.bashrc
echo 'export DB_NAME=solar_panel_db' >> ~/.bashrc
source ~/.bashrc
```

### Connection Pooling
Configure connection pooling in the application:

```python
from sqlalchemy import create_engine
import os

# Get database connection from environment variables
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_USER = os.environ.get('DB_USER', 'solar_user')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'your_secure_password')
DB_NAME = os.environ.get('DB_NAME', 'solar_panel_db')
db_connection_str = f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}'

# Create engine with connection pooling
engine = create_engine(
    db_connection_str,
    pool_size=10,           # Number of connections to keep open
    max_overflow=20,        # Max extra connections when pool is full
    pool_recycle=3600,      # Recycle connections after 1 hour
    pool_pre_ping=True      # Verify connections before use
)
```

## Backup and Recovery

### Automated Backups

#### Using mysqldump
```bash
# Create backup script (backup.sh)
#!/bin/bash
BACKUP_DIR="/path/to/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/solar_panel_db_$TIMESTAMP.sql"

# Create backup
mysqldump -u solar_user -p'your_secure_password' --single-transaction --routines --triggers --events solar_panel_db > $BACKUP_FILE

# Compress backup
gzip $BACKUP_FILE

# Delete backups older than 30 days
find $BACKUP_DIR -name "solar_panel_db_*.sql.gz" -type f -mtime +30 -delete
```

#### Schedule with cron (Linux/macOS)
```bash
# Edit crontab
crontab -e

# Add this line for daily backup at 2 AM
0 2 * * * /path/to/backup.sh
```

#### Schedule with Task Scheduler (Windows)
1. Create a batch file (backup.bat):
   ```batch
   @echo off
   set BACKUP_DIR=C:\path\to\backups
   set TIMESTAMP=%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
   set BACKUP_FILE=%BACKUP_DIR%\solar_panel_db_%TIMESTAMP%.sql
   
   "C:\Program Files\MySQL\MySQL Server 8.0\bin\mysqldump.exe" -u solar_user -p"your_secure_password" --single-transaction --routines --triggers --events solar_panel_db > %BACKUP_FILE%
   
   "C:\Program Files\7-Zip\7z.exe" a %BACKUP_FILE%.gz %BACKUP_FILE%
   del %BACKUP_FILE%
   ```
2. Open Task Scheduler
3. Create a new task to run the batch file daily

### Point-in-Time Recovery

#### Enable Binary Logging
```ini
[mysqld]
log-bin = mysql-bin
binlog_format = ROW
expire_logs_days = 14
```

#### Perform Point-in-Time Recovery
```bash
# Restore the full backup
mysql -u root -p solar_panel_db < full_backup.sql

# Apply binary logs up to a specific point in time
mysqlbinlog --stop-datetime="2023-04-01 12:00:00" mysql-bin.000001 | mysql -u root -p solar_panel_db
```

## Troubleshooting

### Common Issues and Solutions

#### Connection Refused
1. Check if MySQL is running:
   ```bash
   # Windows
   sc query mysql80
   
   # macOS
   ps aux | grep mysql
   
   # Linux
   sudo systemctl status mysql
   ```

2. Check if MySQL is listening on the expected port:
   ```bash
   # Windows
   netstat -ano | findstr 3306
   
   # macOS/Linux
   netstat -tuln | grep 3306
   ```

3. Check firewall settings:
   ```bash
   # Windows
   netsh advfirewall firewall show rule name=all | findstr "MySQL"
   
   # macOS
   sudo /usr/libexec/ApplicationFirewall/socketfilterfw --listapps
   
   # Linux
   sudo ufw status
   ```

#### Authentication Errors
1. Reset root password:
   
   **Windows:**
   ```
   1. Stop MySQL service: net stop mysql80
   2. Start in safe mode: 
      "C:\Program Files\MySQL\MySQL Server 8.0\bin\mysqld.exe" --defaults-file="C:\ProgramData\MySQL\MySQL Server 8.0\my.ini" --init-file=C:\mysql-init.txt --console
      
      Where C:\mysql-init.txt contains:
      ALTER USER 'root'@'localhost' IDENTIFIED BY 'new_password';
   3. Restart normally: net start mysql80
   ```
   
   **Linux:**
   ```
   1. Stop MySQL: sudo systemctl stop mysql
   2. Start in safe mode: sudo mysqld_safe --skip-grant-tables &
   3. Connect: mysql -u root
   4. Reset password:
      UPDATE mysql.user SET authentication_string=PASSWORD('new_password') WHERE User='root' AND Host='localhost';
      FLUSH PRIVILEGES;
   5. Restart: sudo systemctl restart mysql
   ```

2. Check user privileges:
   ```sql
   SHOW GRANTS FOR 'solar_user'@'localhost';
   ```

#### Database Performance Issues
1. Check slow queries:
   ```ini
   [mysqld]
   slow_query_log = 1
   slow_query_log_file = /var/log/mysql/mysql-slow.log
   long_query_time = 2
   ```

2. Analyze queries:
   ```sql
   EXPLAIN SELECT * FROM solar_panel_data WHERE timestamp > '2023-01-01';
   ```

3. Check system resources:
   ```bash
   # CPU and memory usage
   top
   
   # Disk I/O
   iostat -x 1
   ```

4. Check table status:
   ```sql
   SHOW TABLE STATUS FROM solar_panel_db;
   ```

5. Optimize tables:
   ```sql
   OPTIMIZE TABLE solar_panel_data;
   ```

## Advanced Topics

### Replication Setup
For high availability and read scaling:

#### Configure Master Server
```ini
[mysqld]
server-id = 1
log-bin = mysql-bin
binlog_format = ROW
```

#### Configure Replica Server
```ini
[mysqld]
server-id = 2
relay-log = mysql-relay-bin
read_only = ON
```

#### Set Up Replication
On master:
```sql
CREATE USER 'repl_user'@'replica_ip' IDENTIFIED BY 'repl_password';
GRANT REPLICATION SLAVE ON *.* TO 'repl_user'@'replica_ip';
FLUSH TABLES WITH READ LOCK;
SHOW MASTER STATUS;
```

On replica:
```sql
CHANGE MASTER TO
  MASTER_HOST='master_ip',
  MASTER_USER='repl_user',
  MASTER_PASSWORD='repl_password',
  MASTER_LOG_FILE='mysql-bin.000001',
  MASTER_LOG_POS=123;
START SLAVE;
SHOW SLAVE STATUS\G
```

### Partitioning
For large tables like `solar_panel_data`:

```sql
ALTER TABLE solar_panel_data
PARTITION BY RANGE (TO_DAYS(timestamp)) (
    PARTITION p_2023_01 VALUES LESS THAN (TO_DAYS('2023-02-01')),
    PARTITION p_2023_02 VALUES LESS THAN (TO_DAYS('2023-03-01')),
    PARTITION p_2023_03 VALUES LESS THAN (TO_DAYS('2023-04-01')),
    PARTITION p_future VALUES LESS THAN MAXVALUE
);
```

### Monitoring and Alerting
1. Install Prometheus and Grafana
2. Configure MySQL Exporter
3. Set up dashboards for:
   - Connections
   - Query throughput
   - Buffer pool usage
   - Disk I/O
   - Replication lag

### Automating MySQL Administration
Create scripts for common tasks:

#### Health Check Script
```bash
#!/bin/bash
# health_check.sh

# Check MySQL is running
if ! systemctl is-active --quiet mysql; then
    echo "MySQL is not running"
    exit 1
fi

# Check connection
if ! mysql -u solar_user -p'your_secure_password' -e "SELECT 1"; then
    echo "Cannot connect to MySQL"
    exit 1
fi

# Check disk space
DISK_USAGE=$(df -h /var/lib/mysql | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 85 ]; then
    echo "Disk space critical: $DISK_USAGE%"
    exit 1
fi

# Check slow queries
SLOW_QUERIES=$(mysql -u solar_user -p'your_secure_password' -e "SHOW GLOBAL STATUS LIKE 'Slow_queries'" | awk 'NR==2 {print $2}')
echo "Slow queries: $SLOW_QUERIES"

echo "MySQL health check passed"
exit 0
```

---

This comprehensive guide should help you set up and configure MySQL for the Solar Fault Detection System with best practices for security, performance, and reliability. For specific issues or advanced configurations, refer to the [official MySQL documentation](https://dev.mysql.com/doc/).
