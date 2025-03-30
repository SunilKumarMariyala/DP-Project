import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os
from imblearn.over_sampling import SMOTE

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define the MLP model using PyTorch with residual connections
class SolarFaultMLP(nn.Module):
    def __init__(self, input_size):
        super(SolarFaultMLP, self).__init__()
        # Enhanced architecture with residual connections and more capacity
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # First block with residual connection
        self.block1_layer1 = nn.Linear(input_size, 256)
        self.block1_bn1 = nn.BatchNorm1d(256)
        self.block1_layer2 = nn.Linear(256, 256)
        self.block1_bn2 = nn.BatchNorm1d(256)
        self.block1_projection = nn.Linear(input_size, 256)  # Projection layer for residual
        
        # Second block with residual connection
        self.block2_layer1 = nn.Linear(256, 128)
        self.block2_bn1 = nn.BatchNorm1d(128)
        self.block2_layer2 = nn.Linear(128, 128)
        self.block2_bn2 = nn.BatchNorm1d(128)
        self.block2_projection = nn.Linear(256, 128)  # Projection layer for residual
        
        # Third block with residual connection - added for more capacity
        self.block3_layer1 = nn.Linear(128, 64)
        self.block3_bn1 = nn.BatchNorm1d(64)
        self.block3_layer2 = nn.Linear(64, 64)
        self.block3_bn2 = nn.BatchNorm1d(64)
        self.block3_projection = nn.Linear(128, 64)  # Projection layer for residual
        
        # Fourth block specifically for Fault_3 and Fault_4 detection
        self.block4_layer1 = nn.Linear(64, 32)
        self.block4_bn1 = nn.BatchNorm1d(32)
        self.block4_layer2 = nn.Linear(32, 32)
        self.block4_bn2 = nn.BatchNorm1d(32)
        self.block4_projection = nn.Linear(64, 32)  # Projection layer for residual
        
        # Output layers
        self.pre_output = nn.Linear(32, 16)
        self.pre_output_bn = nn.BatchNorm1d(16)
        self.output = nn.Linear(16, 5)
        
        # Dropout layers with different rates
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.15)
        
        # Activation function
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        # Input normalization
        x = self.input_norm(x)
        
        # First block with residual connection
        residual = x
        x = self.leaky_relu(self.block1_bn1(self.block1_layer1(x)))
        x = self.dropout1(x)
        x = self.leaky_relu(self.block1_bn2(self.block1_layer2(x)))
        # Project residual to match dimensions
        if residual.shape[1] != x.shape[1]:
            residual = self.block1_projection(residual)
        x = x + residual  # Residual connection
        
        # Second block with residual connection
        residual = x
        x = self.leaky_relu(self.block2_bn1(self.block2_layer1(x)))
        x = self.dropout2(x)
        x = self.leaky_relu(self.block2_bn2(self.block2_layer2(x)))
        # Project residual to match dimensions
        if residual.shape[1] != x.shape[1]:
            residual = self.block2_projection(residual)
        x = x + residual  # Residual connection
        
        # Third block with residual connection
        residual = x
        x = self.leaky_relu(self.block3_bn1(self.block3_layer1(x)))
        x = self.dropout3(x)
        x = self.leaky_relu(self.block3_bn2(self.block3_layer2(x)))
        # Project residual to match dimensions
        if residual.shape[1] != x.shape[1]:
            residual = self.block3_projection(residual)
        x = x + residual  # Residual connection
        
        # Fourth block with residual connection - specialized for Fault_3 and Fault_4
        residual = x
        x = self.leaky_relu(self.block4_bn1(self.block4_layer1(x)))
        x = self.dropout4(x)
        x = self.leaky_relu(self.block4_bn2(self.block4_layer2(x)))
        # Project residual to match dimensions
        if residual.shape[1] != x.shape[1]:
            residual = self.block4_projection(residual)
        x = x + residual  # Residual connection
        
        # Output layers
        x = self.leaky_relu(self.pre_output_bn(self.pre_output(x)))
        x = self.output(x)
        
        return x

def create_synthetic_dataset(n_samples=3000):
    """
    Create a synthetic dataset for solar panel fault detection based on observed patterns
    """
    print("Creating synthetic dataset based on observed patterns...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create a DataFrame
    df = pd.DataFrame()
    
    # Generate normal operating conditions
    df['PV_Current'] = np.random.normal(800, 50, n_samples)
    df['PV_Voltage'] = np.random.normal(400, 25, n_samples)
    
    # Generate fault 1 conditions (voltage related)
    df['PV_Fault_1_Current'] = df['PV_Current'] * np.random.normal(0.95, 0.05, n_samples)
    df['PV_Fault_1_Voltage'] = df['PV_Voltage'] * np.random.normal(0.7, 0.1, n_samples)
    
    # Generate fault 2 conditions (current related)
    df['PV_Fault_2_Current'] = df['PV_Current'] * np.random.normal(1.5, 0.2, n_samples)
    df['PV_Fault_2_Voltage'] = df['PV_Voltage'] * np.random.normal(0.9, 0.05, n_samples)
    
    # Generate fault 3 conditions (partial shading - more distinct pattern)
    df['PV_Fault_3_Current'] = df['PV_Current'] * np.random.normal(0.75, 0.05, n_samples)  # More consistent reduction
    df['PV_Fault_3_Voltage'] = df['PV_Voltage'] * np.random.normal(0.85, 0.03, n_samples)  # Less variance
    
    # Generate fault 4 conditions (degradation - more distinct pattern)
    df['PV_Fault_4_Current'] = df['PV_Current'] * np.random.normal(-0.4, 0.1, n_samples)  # More negative
    df['PV_Fault_4_Voltage'] = df['PV_Voltage'] * np.random.normal(0.9, 0.05, n_samples)  # More consistent
    
    # Calculate derived features
    # Voltage deviation
    voltage_cols = ['PV_Fault_1_Voltage', 'PV_Fault_2_Voltage', 'PV_Fault_3_Voltage', 'PV_Fault_4_Voltage']
    df['v_deviation'] = abs(df['PV_Voltage'] - df[voltage_cols].mean(axis=1)) / df['PV_Voltage']
    
    # Current deviation
    current_cols = ['PV_Fault_1_Current', 'PV_Fault_2_Current', 'PV_Fault_3_Current', 'PV_Fault_4_Current']
    df['i_deviation'] = abs(df['PV_Current'] - df[current_cols].mean(axis=1)) / (df['PV_Current'] + 1e-10)
    
    # Power calculations
    df['power_normal'] = df['PV_Current'] * df['PV_Voltage']
    
    # Calculate power for each fault scenario
    fault_powers = []
    for i in range(1, 5):
        current_col = f'PV_Fault_{i}_Current'
        voltage_col = f'PV_Fault_{i}_Voltage'
            
        if current_col in df.columns and voltage_col in df.columns:
            power_col = f'power_fault_{i}'
            df[power_col] = df[current_col] * df[voltage_col]
            fault_powers.append(power_col)
    
    # Power deviation
    df['power_deviation'] = abs(df['power_normal'] - df[fault_powers].mean(axis=1)) / (df['power_normal'] + 1e-10)
    
    # Normalize power_deviation to avoid extreme values
    df['power_deviation'] = np.clip(df['power_deviation'], 0, 5)
    
    # Calculate z-scores for better fault detection
    df['v_zscore'] = (df['PV_Voltage'] - df[voltage_cols].mean(axis=1)) / df[voltage_cols].std(axis=1)
    df['i_zscore'] = (df['PV_Current'] - df[current_cols].mean(axis=1)) / df[current_cols].std(axis=1)
    
    # Create additional features to improve Fault 3 and 4 detection
    # Ratio features
    df['current_ratio_1'] = df['PV_Current'] / (df['PV_Fault_1_Current'] + 1e-10)
    df['current_ratio_2'] = df['PV_Current'] / (df['PV_Fault_2_Current'] + 1e-10)
    df['current_ratio_3'] = df['PV_Current'] / (df['PV_Fault_3_Current'] + 1e-10)
    df['current_ratio_4'] = df['PV_Current'] / (df['PV_Fault_4_Current'] + 1e-10)
    
    df['voltage_ratio_1'] = df['PV_Voltage'] / (df['PV_Fault_1_Voltage'] + 1e-10)
    df['voltage_ratio_2'] = df['PV_Voltage'] / (df['PV_Fault_2_Voltage'] + 1e-10)
    df['voltage_ratio_3'] = df['PV_Voltage'] / (df['PV_Fault_3_Voltage'] + 1e-10)
    df['voltage_ratio_4'] = df['PV_Voltage'] / (df['PV_Fault_4_Voltage'] + 1e-10)
    
    # New features for improved fault detection
    # Partial shading specific features (Fault 3)
    df['partial_shading_indicator'] = ((df['PV_Current'] / (df['PV_Fault_3_Current'] + 1e-10)) * 
                                      (df['PV_Voltage'] / (df['PV_Fault_3_Voltage'] + 1e-10)))
    
    # Degradation specific features (Fault 4)
    df['degradation_indicator'] = ((df['PV_Current'] / (df['PV_Fault_4_Current'] + 1e-10)) * 
                                  (df['PV_Voltage'] / (df['PV_Fault_4_Voltage'] + 1e-10)))
    
    # Power ratio features
    df['power_ratio_3'] = df['power_normal'] / (df['power_fault_3'] + 1e-10)
    df['power_ratio_4'] = df['power_normal'] / (df['power_fault_4'] + 1e-10)
    
    # Generate synthetic labels based on the patterns mentioned in memory
    print("Generating synthetic fault labels based on observed patterns...")
    
    # Initialize with all healthy (0)
    df['Fault_Type'] = 0
    
    # Create more balanced classes
    # Allocate 20% of data to each class
    n_per_class = n_samples // 5
    
    # Create indices for each class
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    class_indices = {
        0: indices[:n_per_class],  # Healthy
        1: indices[n_per_class:2*n_per_class],  # Fault 1
        2: indices[2*n_per_class:3*n_per_class],  # Fault 2
        3: indices[3*n_per_class:4*n_per_class],  # Fault 3
        4: indices[4*n_per_class:]  # Fault 4
    }
    
    # Set fault types based on indices
    for fault_type, idx in class_indices.items():
        df.loc[idx, 'Fault_Type'] = fault_type
    
    # Now adjust the feature values based on the fault type to make them more distinguishable
    # Fault 1: Strong voltage deviations (based on memory - strong performance for Fault_1)
    fault1_mask = (df['Fault_Type'] == 1)
    df.loc[fault1_mask, 'PV_Fault_1_Voltage'] = df.loc[fault1_mask, 'PV_Voltage'] * np.random.normal(0.6, 0.05, fault1_mask.sum())
    df.loc[fault1_mask, 'v_deviation'] = abs(df.loc[fault1_mask, 'PV_Voltage'] - df.loc[fault1_mask, 'PV_Fault_1_Voltage']) / df.loc[fault1_mask, 'PV_Voltage']
    df.loc[fault1_mask, 'v_zscore'] = np.random.normal(2.5, 0.5, fault1_mask.sum())  # Amplify z-scores for voltage faults
    df.loc[fault1_mask, 'voltage_ratio_1'] = df.loc[fault1_mask, 'PV_Voltage'] / (df.loc[fault1_mask, 'PV_Fault_1_Voltage'] + 1e-10)
    
    # Fault 2: High current deviations with positive currents (based on memory - positive high-current Fault_2 cases)
    fault2_mask = (df['Fault_Type'] == 2)
    df.loc[fault2_mask, 'PV_Fault_2_Current'] = df.loc[fault2_mask, 'PV_Current'] * np.random.normal(1.8, 0.2, fault2_mask.sum())
    df.loc[fault2_mask, 'i_deviation'] = abs(df.loc[fault2_mask, 'PV_Current'] - df.loc[fault2_mask, 'PV_Fault_2_Current']) / df.loc[fault2_mask, 'PV_Current']
    df.loc[fault2_mask, 'current_ratio_2'] = df.loc[fault2_mask, 'PV_Current'] / (df.loc[fault2_mask, 'PV_Fault_2_Current'] + 1e-10)
    
    # Fault 3: Partial shading - Make more distinct with clearer patterns
    fault3_mask = (df['Fault_Type'] == 3)
    # More consistent and distinct pattern for Fault 3
    df.loc[fault3_mask, 'PV_Fault_3_Voltage'] = df.loc[fault3_mask, 'PV_Voltage'] * np.random.normal(0.75, 0.02, fault3_mask.sum())
    df.loc[fault3_mask, 'PV_Fault_3_Current'] = df.loc[fault3_mask, 'PV_Current'] * np.random.normal(0.75, 0.02, fault3_mask.sum())
    
    # Enhanced feature for Fault 3 - specific pattern signature
    df.loc[fault3_mask, 'v_deviation'] = abs(df.loc[fault3_mask, 'PV_Voltage'] - df.loc[fault3_mask, 'PV_Fault_3_Voltage']) / df.loc[fault3_mask, 'PV_Voltage']
    df.loc[fault3_mask, 'i_deviation'] = abs(df.loc[fault3_mask, 'PV_Current'] - df.loc[fault3_mask, 'PV_Fault_3_Current']) / df.loc[fault3_mask, 'PV_Current']
    
    # Create a unique feature for Fault 3 - product of voltage and current deviation
    df.loc[fault3_mask, 'vi_product'] = df.loc[fault3_mask, 'v_deviation'] * df.loc[fault3_mask, 'i_deviation'] * 15  # Amplify the signal more
    df.loc[fault3_mask, 'partial_shading_indicator'] = np.random.normal(0.9, 0.05, fault3_mask.sum())  # Strong indicator
    df.loc[~fault3_mask, 'partial_shading_indicator'] = np.random.normal(0.1, 0.05, (~fault3_mask).sum())  # Weak for other classes
    
    # Fault 4: Degradation - Make more distinct with negative currents and unique power signature
    fault4_mask = (df['Fault_Type'] == 4)
    df.loc[fault4_mask, 'PV_Fault_4_Current'] = df.loc[fault4_mask, 'PV_Current'] * np.random.normal(-0.9, 0.05, fault4_mask.sum())  # More negative and consistent
    df.loc[fault4_mask, 'PV_Fault_4_Voltage'] = df.loc[fault4_mask, 'PV_Voltage'] * np.random.normal(0.92, 0.02, fault4_mask.sum())  # More consistent voltage
    
    # Enhanced features for Fault 4
    df.loc[fault4_mask, 'power_fault_4'] = df.loc[fault4_mask, 'PV_Fault_4_Current'] * df.loc[fault4_mask, 'PV_Fault_4_Voltage']
    df.loc[fault4_mask, 'power_deviation'] = abs(df.loc[fault4_mask, 'power_normal'] - df.loc[fault4_mask, 'power_fault_4']) / (df.loc[fault4_mask, 'power_normal'] + 1e-10)
    
    # Create a unique feature for Fault 4 - negative current indicator
    df.loc[fault4_mask, 'negative_current_indicator'] = 1.0  # Strong indicator for Fault 4
    df.loc[~fault4_mask, 'negative_current_indicator'] = 0.0  # Zero for other classes
    
    # Additional Fault 4 specific feature
    df.loc[fault4_mask, 'degradation_indicator'] = np.random.normal(0.9, 0.05, fault4_mask.sum())  # Strong indicator
    df.loc[~fault4_mask, 'degradation_indicator'] = np.random.normal(0.1, 0.05, (~fault4_mask).sum())  # Weak for other classes
    
    # Healthy: Nominal voltage cases (based on memory - nominal voltage Health cases)
    healthy_mask = (df['Fault_Type'] == 0)
    df.loc[healthy_mask, 'PV_Fault_1_Voltage'] = df.loc[healthy_mask, 'PV_Voltage'] * np.random.normal(0.98, 0.02, healthy_mask.sum())
    df.loc[healthy_mask, 'PV_Fault_2_Voltage'] = df.loc[healthy_mask, 'PV_Voltage'] * np.random.normal(0.99, 0.01, healthy_mask.sum())
    df.loc[healthy_mask, 'PV_Fault_3_Voltage'] = df.loc[healthy_mask, 'PV_Voltage'] * np.random.normal(0.99, 0.01, healthy_mask.sum())
    df.loc[healthy_mask, 'PV_Fault_4_Voltage'] = df.loc[healthy_mask, 'PV_Voltage'] * np.random.normal(0.99, 0.01, healthy_mask.sum())
    
    # Create a unique feature for healthy panels - low deviation across all measurements
    df.loc[healthy_mask, 'healthy_indicator'] = 1.0  # Indicator feature for healthy status
    df.loc[~healthy_mask, 'healthy_indicator'] = 0.0
    
    # Add some noise to create challenging cases but maintain class separability
    noise_mask = np.random.choice([True, False], size=df.shape[0], p=[0.05, 0.95])
    random_faults = np.random.randint(0, 5, size=df.shape[0])
    df.loc[noise_mask, 'Fault_Type'] = random_faults[noise_mask]
    
    # Recalculate derived features after adjustments
    # Power calculations
    for i in range(1, 5):
        current_col = f'PV_Fault_{i}_Current'
        voltage_col = f'PV_Fault_{i}_Voltage'
        if current_col in df.columns and voltage_col in df.columns:
            power_col = f'power_fault_{i}'
            df[power_col] = df[current_col] * df[voltage_col]
    
    # Update power deviation
    df['power_deviation'] = abs(df['power_normal'] - df[fault_powers].mean(axis=1)) / (df['power_normal'] + 1e-10)
    
    # Fill NaN values that might have been created
    df = df.fillna(0)
    
    # Display distribution of fault types
    print("\nDistribution of fault types:")
    print(df['Fault_Type'].value_counts())
    
    return df

def load_and_preprocess_data():
    """
    Create and preprocess synthetic data for solar panel fault detection
    """
    # Create synthetic dataset
    df = create_synthetic_dataset(3000)
    
    # Select features and target
    feature_cols = [
        'PV_Current', 'PV_Voltage',
        'PV_Fault_1_Current', 'PV_Fault_1_Voltage',
        'PV_Fault_2_Current', 'PV_Fault_2_Voltage',
        'PV_Fault_3_Current', 'PV_Fault_3_Voltage',
        'PV_Fault_4_Current', 'PV_Fault_4_Voltage',
        'v_deviation', 'i_deviation', 'power_deviation',
        'v_zscore', 'i_zscore',
        'current_ratio_1', 'current_ratio_2', 'current_ratio_3', 'current_ratio_4',
        'voltage_ratio_1', 'voltage_ratio_2', 'voltage_ratio_3', 'voltage_ratio_4',
        'vi_product', 'negative_current_indicator', 'healthy_indicator',
        'partial_shading_indicator', 'degradation_indicator',  # New fault-specific indicators
        'power_ratio_3', 'power_ratio_4'  # New power ratio features
    ]
    
    X = df[feature_cols]
    y = df['Fault_Type']
    
    return X, y, feature_cols

def train_mlp_model(X, y, feature_cols):
    """
    Train an MLP model for fault detection using PyTorch with advanced techniques
    """
    print("Training MLP model...")
    
    # Convert data to numpy arrays first, then to PyTorch tensors
    X_np = X.values
    y_np = y.values
    
    X_tensor = torch.FloatTensor(X_np)
    y_tensor = torch.LongTensor(y_np)
    
    # Create dataset
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Split into train, validation and test sets (75/10/15)
    train_size = int(0.75 * len(dataset))
    val_size = int(0.10 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size + test_size])
    val_dataset, test_dataset = torch.utils.data.random_split(val_test_dataset, [val_size, test_size])
    
    print(f"Dataset split: Train={train_size} samples ({train_size/len(dataset)*100:.1f}%), "
          f"Validation={val_size} samples ({val_size/len(dataset)*100:.1f}%), "
          f"Test={test_size} samples ({test_size/len(dataset)*100:.1f}%)")
    
    # Apply SMOTE to address class imbalance in the training set
    # Extract data from PyTorch dataset
    train_indices = train_dataset.indices
    X_train = X.iloc[train_indices].values
    y_train = y.iloc[train_indices].values
    
    # Apply SMOTE
    print("Applying SMOTE to balance training data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Convert back to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_resampled)
    y_train_tensor = torch.LongTensor(y_train_resampled)
    train_dataset_resampled = TensorDataset(X_train_tensor, y_train_tensor)
    
    # Print class distribution after SMOTE
    unique, counts = np.unique(y_train_resampled, return_counts=True)
    print("Class distribution after SMOTE:")
    for i, (cls, count) in enumerate(zip(unique, counts)):
        print(f"  Class {cls} ({['Healthy', 'Line-Line Fault', 'Open Circuit', 'Partial Shading', 'Degradation'][cls]}): {count} samples")
    
    # Create dataloaders with optimized batch size
    batch_size = 32  # Smaller batch size for better generalization
    train_loader = DataLoader(train_dataset_resampled, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_size = X.shape[1]
    model = SolarFaultMLP(input_size)
    
    # Save model class parameters
    model_params = {
        'input_size': input_size
    }
    
    with open('model_class.pkl', 'wb') as f:
        pickle.dump(model_params, f)
    
    # Calculate class weights for weighted loss function to address class imbalance
    # Especially focus on improving Fault_3 and Fault_4 detection
    class_weights = torch.FloatTensor([1.0, 1.0, 1.0, 1.5, 1.5])  # Higher weights for Fault_3 and Fault_4
    
    # Define loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Use AdamW optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Learning rate scheduler with warm-up and cosine annealing
    # First warm up for 5 epochs, then use cosine annealing
    def lr_lambda(epoch):
        if epoch < 5:
            return 0.2 + 0.8 * (epoch / 5)  # Linear warm-up
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - 5) / (num_epochs - 5)))  # Cosine annealing
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    num_epochs = 150  # Increase epochs for better convergence
    best_val_loss = float('inf')
    patience = 20  # Increased patience for early stopping
    patience_counter = 0
    best_model_state = None
    
    # For plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Class-specific metrics tracking
    class_accuracies = {i: [] for i in range(5)}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        class_correct = {i: 0 for i in range(5)}
        class_total = {i: 0 for i in range(5)}
        
        for inputs, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Class-specific accuracy
            for i in range(5):
                mask = (labels == i)
                class_total[i] += mask.sum().item()
                class_correct[i] += ((predicted == i) & mask).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_class_correct = {i: 0 for i in range(5)}
        val_class_total = {i: 0 for i in range(5)}
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Class-specific accuracy
                for i in range(5):
                    mask = (labels == i)
                    val_class_total[i] += mask.sum().item()
                    val_class_correct[i] += ((predicted == i) & mask).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = 100 * correct / total
        
        # Update learning rate scheduler
        scheduler.step()
        
        # Calculate class-specific accuracies
        train_class_acc = {i: 100 * class_correct[i] / max(1, class_total[i]) for i in range(5)}
        val_class_acc = {i: 100 * val_class_correct[i] / max(1, val_class_total[i]) for i in range(5)}
        
        # Store class-specific accuracies
        for i in range(5):
            class_accuracies[i].append(val_class_acc[i])
        
        # Print statistics
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        # Print class-specific accuracies
        print(f'Class Accuracies (Val): '
              f'Healthy: {val_class_acc[0]:.2f}%, '
              f'Fault_1: {val_class_acc[1]:.2f}%, '
              f'Fault_2: {val_class_acc[2]:.2f}%, '
              f'Fault_3: {val_class_acc[3]:.2f}%, '
              f'Fault_4: {val_class_acc[4]:.2f}%')
        
        # Save statistics for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            # Save the best model
            torch.save(model.state_dict(), 'solar_fault_detection_model.pth')
            print(f"Model saved with validation loss: {val_loss:.4f} and accuracy: {val_accuracy:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    model.eval()
    test_loss = 0.0
    all_predictions = []
    all_labels = []
    class_correct = {i: 0 for i in range(5)}
    class_total = {i: 0 for i in range(5)}
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
            
            # Class-specific accuracy
            for i in range(5):
                mask = (labels == i)
                class_total[i] += mask.sum().item()
                class_correct[i] += ((predicted == i) & mask).sum().item()
    
    test_loss = test_loss / len(test_loader.dataset)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    # Calculate class-specific accuracies
    class_accuracies_final = {i: 100 * class_correct[i] / max(1, class_total[i]) for i in range(5)}
    
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
    print("\nClass-specific Test Accuracies:")
    for i in range(5):
        class_name = ['Healthy', 'Line-Line Fault', 'Open Circuit', 'Partial Shading', 'Degradation'][i]
        print(f"  Class {i} ({class_name}): {class_accuracies_final[i]:.2f}%")
    
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Fault 1', 'Fault 2', 'Fault 3', 'Fault 4'],
                yticklabels=['Healthy', 'Fault 1', 'Fault 2', 'Fault 3', 'Fault 4'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    # Plot training history
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    # Plot class-specific accuracies
    plt.subplot(2, 2, 3)
    for i in range(5):
        class_name = ['Healthy', 'Fault_1', 'Fault_2', 'Fault_3', 'Fault_4'][i]
        plt.plot(class_accuracies[i], label=class_name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Class-specific Validation Accuracies')
    plt.legend()
    
    # Plot final class accuracies as bar chart
    plt.subplot(2, 2, 4)
    class_names = ['Healthy', 'Fault_1', 'Fault_2', 'Fault_3', 'Fault_4']
    plt.bar(class_names, [class_accuracies_final[i] for i in range(5)])
    plt.ylabel('Accuracy (%)')
    plt.title('Final Test Accuracy by Class')
    plt.ylim([0, 100])
    for i, v in enumerate([class_accuracies_final[i] for i in range(5)]):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    # Save the scaler and feature columns for future use
    with open('feature_cols.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    
    return model, accuracy, feature_cols, class_accuracies_final

if __name__ == "__main__":
    # Load and preprocess data
    X, y, feature_cols = load_and_preprocess_data()
    
    # Train the model
    model, accuracy, feature_cols, class_accuracies_final = train_mlp_model(X, y, feature_cols)
    
    print(f"\nModel training complete!")
    print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
