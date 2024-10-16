import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(filepath):
    # Load data
    df = pd.read_csv(filepath)
    
    # Example preprocessing steps
    df.fillna(df.mean(), inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    
    # Split features and labels
    X = df.drop('classification', axis=1)
    y = df['classification']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save preprocessed data
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)

if __name__ == "__main__":
    preprocess_data('data/raw/kidney.csv')
