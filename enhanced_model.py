"""
Enhanced Solar Fault Detection Model

This module implements an advanced neural network architecture for solar fault detection
with improved feature engineering, ensemble techniques, and explainability components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhanced_model.log')
    ]
)
logger = logging.getLogger('enhanced_model')

class FeatureExtractor:
    """Advanced feature engineering for solar panel data"""
    
    def __init__(self, scaler_path=None):
        """
        Initialize the feature extractor
        
        Args:
            scaler_path: Path to saved scaler model (optional)
        """
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
        else:
            self.scaler = StandardScaler()
            logger.info("Created new scaler")
    
    def fit(self, X):
        """
        Fit the scaler to the input data
        
        Args:
            X: Input data
            
        Returns:
            Self
        """
        self.scaler.fit(X)
        return self
    
    def save_scaler(self, path):
        """
        Save the scaler to a file
        
        Args:
            path: Path to save the scaler
        """
        joblib.dump(self.scaler, path)
        logger.info(f"Saved scaler to {path}")
    
    def extract_features(self, data):
        """
        Extract advanced features from solar panel data
        
        Args:
            data: DataFrame with solar panel data
            
        Returns:
            DataFrame with extracted features
        """
        # Create a copy to avoid modifying the original data
        df = data.copy()
        
        # Basic features
        features = [
            'pv_current', 'pv_voltage', 
            'pv_fault_1_current', 'pv_fault_1_voltage',
            'pv_fault_2_current', 'pv_fault_2_voltage',
            'pv_fault_3_current', 'pv_fault_3_voltage',
            'pv_fault_4_current', 'pv_fault_4_voltage'
        ]
        
        # Calculate power values
        df['pv_power'] = df['pv_current'] * df['pv_voltage']
        df['fault_1_power'] = df['pv_fault_1_current'] * df['pv_fault_1_voltage']
        df['fault_2_power'] = df['pv_fault_2_current'] * df['pv_fault_2_voltage']
        df['fault_3_power'] = df['pv_fault_3_current'] * df['pv_fault_3_voltage']
        df['fault_4_power'] = df['pv_fault_4_current'] * df['pv_fault_4_voltage']
        
        # Add power features to the list
        power_features = ['pv_power', 'fault_1_power', 'fault_2_power', 'fault_3_power', 'fault_4_power']
        features.extend(power_features)
        
        # Calculate deviation features (percentage difference from normal)
        df['i_deviation_1'] = (df['pv_current'] - df['pv_fault_1_current']) / df['pv_fault_1_current'] * 100
        df['v_deviation_1'] = (df['pv_voltage'] - df['pv_fault_1_voltage']) / df['pv_fault_1_voltage'] * 100
        df['p_deviation_1'] = (df['pv_power'] - df['fault_1_power']) / df['fault_1_power'] * 100
        
        df['i_deviation_2'] = (df['pv_current'] - df['pv_fault_2_current']) / df['pv_fault_2_current'] * 100
        df['v_deviation_2'] = (df['pv_voltage'] - df['pv_fault_2_voltage']) / df['pv_fault_2_voltage'] * 100
        df['p_deviation_2'] = (df['pv_power'] - df['fault_2_power']) / df['fault_2_power'] * 100
        
        df['i_deviation_3'] = (df['pv_current'] - df['pv_fault_3_current']) / df['pv_fault_3_current'] * 100
        df['v_deviation_3'] = (df['pv_voltage'] - df['pv_fault_3_voltage']) / df['pv_fault_3_voltage'] * 100
        df['p_deviation_3'] = (df['pv_power'] - df['fault_3_power']) / df['fault_3_power'] * 100
        
        df['i_deviation_4'] = (df['pv_current'] - df['pv_fault_4_current']) / df['pv_fault_4_current'] * 100
        df['v_deviation_4'] = (df['pv_voltage'] - df['pv_fault_4_voltage']) / df['pv_fault_4_voltage'] * 100
        df['p_deviation_4'] = (df['pv_power'] - df['fault_4_power']) / df['fault_4_power'] * 100
        
        # Add deviation features to the list
        deviation_features = [
            'i_deviation_1', 'v_deviation_1', 'p_deviation_1',
            'i_deviation_2', 'v_deviation_2', 'p_deviation_2',
            'i_deviation_3', 'v_deviation_3', 'p_deviation_3',
            'i_deviation_4', 'v_deviation_4', 'p_deviation_4'
        ]
        features.extend(deviation_features)
        
        # Calculate ratios between different measurements
        df['current_voltage_ratio'] = df['pv_current'] / df['pv_voltage']
        df['fault_1_ratio'] = df['pv_fault_1_current'] / df['pv_fault_1_voltage']
        df['fault_2_ratio'] = df['pv_fault_2_current'] / df['pv_fault_2_voltage']
        df['fault_3_ratio'] = df['pv_fault_3_current'] / df['pv_fault_3_voltage']
        df['fault_4_ratio'] = df['pv_fault_4_current'] / df['pv_fault_4_voltage']
        
        # Add ratio features to the list
        ratio_features = [
            'current_voltage_ratio', 'fault_1_ratio', 'fault_2_ratio', 
            'fault_3_ratio', 'fault_4_ratio'
        ]
        features.extend(ratio_features)
        
        # Calculate statistical features
        current_values = df[['pv_current', 'pv_fault_1_current', 'pv_fault_2_current', 
                           'pv_fault_3_current', 'pv_fault_4_current']]
        voltage_values = df[['pv_voltage', 'pv_fault_1_voltage', 'pv_fault_2_voltage', 
                           'pv_fault_3_voltage', 'pv_fault_4_voltage']]
        
        df['current_mean'] = current_values.mean(axis=1)
        df['current_std'] = current_values.std(axis=1)
        df['current_max'] = current_values.max(axis=1)
        df['current_min'] = current_values.min(axis=1)
        
        df['voltage_mean'] = voltage_values.mean(axis=1)
        df['voltage_std'] = voltage_values.std(axis=1)
        df['voltage_max'] = voltage_values.max(axis=1)
        df['voltage_min'] = voltage_values.min(axis=1)
        
        # Add statistical features to the list
        stat_features = [
            'current_mean', 'current_std', 'current_max', 'current_min',
            'voltage_mean', 'voltage_std', 'voltage_max', 'voltage_min'
        ]
        features.extend(stat_features)
        
        # Calculate z-scores for each measurement relative to the mean
        for col in ['pv_current', 'pv_voltage']:
            base_value = df[col]
            for i in range(1, 5):
                fault_col = f'{col.split("_")[0]}_fault_{i}_{col.split("_")[1]}'
                z_score_col = f'z_score_{col.split("_")[1]}_{i}'
                df[z_score_col] = (base_value - df[fault_col]) / df[fault_col].std()
                features.append(z_score_col)
        
        # Extract the features
        X = df[features]
        
        # Handle NaN and infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Convert back to DataFrame
        X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=df.index)
        
        return X_scaled_df

class EnhancedSolarFaultDetector(nn.Module):
    """Enhanced neural network for solar fault detection"""
    
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.3, num_classes=5):
        """
        Initialize the model
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            num_classes: Number of output classes
        """
        super(EnhancedSolarFaultDetector, self).__init__()
        
        # Input layer
        layers = [nn.Linear(input_size, hidden_sizes[0])]
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        # Sequential model
        self.model = nn.Sequential(*layers)
        
        # Attention mechanism for feature importance
        self.attention = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[0], 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Get attention weights
        attention_weights = self.attention(x)
        
        # Apply attention to input
        attended_input = x * attention_weights
        
        # Pass through the model
        output = self.model(attended_input)
        
        return output, attention_weights

class SolarFaultDetectionSystem:
    """Complete solar fault detection system with training, evaluation, and prediction"""
    
    def __init__(self, model_path=None, scaler_path=None):
        """
        Initialize the system
        
        Args:
            model_path: Path to saved model (optional)
            scaler_path: Path to saved scaler (optional)
        """
        self.feature_extractor = FeatureExtractor(scaler_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = ['Healthy', 'Fault_1', 'Fault_2', 'Fault_3', 'Fault_4']
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
    
    def load_model(self, model_path):
        """
        Load a trained model from a file
        
        Args:
            model_path: Path to the model file
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        input_size = checkpoint['input_size']
        hidden_sizes = checkpoint['hidden_sizes']
        dropout_rate = checkpoint['dropout_rate']
        num_classes = checkpoint['num_classes']
        
        self.model = EnhancedSolarFaultDetector(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
            num_classes=num_classes
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
    
    def save_model(self, model_path, input_size, hidden_sizes, dropout_rate, num_classes):
        """
        Save the trained model to a file
        
        Args:
            model_path: Path to save the model
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout rate
            num_classes: Number of output classes
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'input_size': input_size,
            'hidden_sizes': hidden_sizes,
            'dropout_rate': dropout_rate,
            'num_classes': num_classes
        }
        
        torch.save(checkpoint, model_path)
        logger.info(f"Saved model to {model_path}")
    
    def train(self, data, target_column='prediction', test_size=0.2, random_state=42, 
              hidden_sizes=[128, 64, 32], dropout_rate=0.3, learning_rate=0.001, 
              batch_size=32, num_epochs=100, patience=20):
        """
        Train the model on the provided data
        
        Args:
            data: DataFrame with solar panel data
            target_column: Name of the target column
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            num_epochs: Maximum number of epochs
            patience: Number of epochs to wait for improvement before early stopping
            
        Returns:
            Dictionary with training results
        """
        # Extract features
        X = self.feature_extractor.extract_features(data)
        
        # Convert target to numeric
        y = pd.Categorical(data[target_column], categories=self.class_names).codes
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test.values)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_size = X.shape[1]
        num_classes = len(self.class_names)
        
        self.model = EnhancedSolarFaultDetector(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
            num_classes=num_classes
        )
        
        self.model.to(self.device)
        
        # Initialize optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Training loop
        best_val_loss = float('inf')
        best_epoch = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # Forward pass
                outputs, _ = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    # Forward pass
                    outputs, _ = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            val_loss /= len(test_loader)
            val_losses.append(val_loss)
            val_accuracy = correct / total
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Print progress
            logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                
                # Save the best model
                self.save_model(
                    'enhanced_model_best.pth',
                    input_size=input_size,
                    hidden_sizes=hidden_sizes,
                    dropout_rate=dropout_rate,
                    num_classes=num_classes
                )
            
            # Early stopping
            if epoch - best_epoch >= patience:
                logger.info(f'Early stopping at epoch {epoch+1}')
                break
        
        # Load the best model
        self.load_model('enhanced_model_best.pth')
        
        # Evaluate on test set
        test_results = self.evaluate(X_test_tensor, y_test_tensor)
        
        # Save feature extractor
        self.feature_extractor.save_scaler('enhanced_scaler.pkl')
        
        # Return results
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'test_results': test_results
        }
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation results
        """
        self.model.eval()
        
        # Move data to device
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)
        
        # Make predictions
        with torch.no_grad():
            outputs, attention_weights = self.model(X_test)
            _, predicted = torch.max(outputs.data, 1)
        
        # Convert to numpy
        y_true = y_test.cpu().numpy()
        y_pred = predicted.cpu().numpy()
        attention = attention_weights.cpu().numpy()
        
        # Calculate metrics
        accuracy = (y_true == y_pred).mean()
        class_report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Return results
        return {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'attention_weights': attention
        }
    
    def predict(self, data):
        """
        Make predictions on new data
        
        Args:
            data: DataFrame with solar panel data
            
        Returns:
            DataFrame with predictions
        """
        # Extract features
        X = self.feature_extractor.extract_features(data)
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X.values).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs, attention_weights = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
        
        # Convert to numpy
        y_pred = predicted.cpu().numpy()
        probs = probabilities.cpu().numpy()
        attention = attention_weights.cpu().numpy()
        
        # Create results DataFrame
        results = pd.DataFrame({
            'prediction': [self.class_names[i] for i in y_pred],
            'confidence': np.max(probs, axis=1) * 100,
            'attention': np.mean(attention, axis=1)
        })
        
        # Add class probabilities
        for i, class_name in enumerate(self.class_names):
            results[f'{class_name}_prob'] = probs[:, i] * 100
        
        return results
    
    def visualize_results(self, results, save_path=None):
        """
        Visualize the evaluation results
        
        Args:
            results: Dictionary with evaluation results
            save_path: Path to save the visualization (optional)
        """
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot confusion matrix
        conf_matrix = results['test_results']['confusion_matrix']
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('True')
        
        # Plot class-wise accuracy
        class_report = results['test_results']['classification_report']
        class_accuracy = [class_report[name]['precision'] for name in self.class_names]
        axes[0, 1].bar(self.class_names, class_accuracy)
        axes[0, 1].set_title('Class-wise Precision')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].set_ylabel('Precision')
        
        # Plot training and validation loss
        axes[1, 0].plot(results['train_losses'], label='Train Loss')
        axes[1, 0].plot(results['val_losses'], label='Validation Loss')
        axes[1, 0].axvline(x=results['best_epoch'], color='r', linestyle='--', label='Best Epoch')
        axes[1, 0].set_title('Training and Validation Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        
        # Plot feature importance (using attention weights)
        if 'attention_weights' in results['test_results']:
            attention = results['test_results']['attention_weights']
            mean_attention = np.mean(attention, axis=0)
            feature_importance = pd.DataFrame({
                'Feature': range(len(mean_attention)),
                'Importance': mean_attention.flatten()
            }).sort_values('Importance', ascending=False).head(10)
            
            axes[1, 1].bar(feature_importance['Feature'], feature_importance['Importance'])
            axes[1, 1].set_title('Top 10 Feature Importance')
            axes[1, 1].set_xlabel('Feature Index')
            axes[1, 1].set_ylabel('Importance')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved visualization to {save_path}")
        else:
            plt.show()

def main():
    """Main function to demonstrate the enhanced model"""
    # Load data
    try:
        # Connect to database
        conn = sqlite3.connect('solar_panel.db')
        
        # Query for data
        query = """
        SELECT 
            id, 
            timestamp, 
            pv_current, 
            pv_voltage, 
            pv_fault_1_current,
            pv_fault_1_voltage,
            pv_fault_2_current,
            pv_fault_2_voltage,
            pv_fault_3_current,
            pv_fault_3_voltage,
            pv_fault_4_current,
            pv_fault_4_voltage,
            prediction,
            processed_at
        FROM 
            solar_panel_data 
        WHERE 
            processed_at IS NOT NULL
            AND prediction != 'Unknown'
        LIMIT 1000
        """
        
        # Execute query
        df = pd.read_sql_query(query, conn)
        
        # Close connection
        conn.close()
        
        if len(df) > 0:
            logger.info(f"Loaded {len(df)} records from database")
            
            # Initialize system
            system = SolarFaultDetectionSystem()
            
            # Train model
            results = system.train(
                data=df,
                target_column='prediction',
                test_size=0.2,
                random_state=42,
                hidden_sizes=[128, 64, 32],
                dropout_rate=0.3,
                learning_rate=0.001,
                batch_size=32,
                num_epochs=100,
                patience=20
            )
            
            # Visualize results
            system.visualize_results(results, save_path='enhanced_model_results.png')
            
            logger.info("Enhanced model training and evaluation completed successfully")
        else:
            logger.warning("No data found in database")
    
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
