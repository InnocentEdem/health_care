from data_loader import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Load the dataset
dataset_name = "shriyashjagtap/heart-attack-risk-assessment-dataset"
df = load_dataset(dataset_name)

# 2. Check if the dataset was loaded successfully
if df is not None:
    print("Dataset Preview:")
    print(df.head())

    # STEP 1: Calculate Average Vitals
    # We'll compute the mean of age, systolic_bp, and total_cholesterol.
    average_vitals = df[['age', 'systolic_bp', 'total_cholesterol']].mean()
    print("\nAverage Vitals:")
    print(average_vitals)

    # STEP 2: Identify Abnormal Readings
    # We'll define "abnormal" for blood pressure if:
    # - systolic_bp > 140 or diastolic_bp > 90 (hypertension threshold)
    # For cholesterol, we'll say total_cholesterol > 200 is high.

    abnormal_bp = df[(df['systolic_bp'] > 140) | (df['diastolic_bp'] > 90)]
    abnormal_chol = df[df['total_cholesterol'] > 200]

    print("\nAbnormal Blood Pressure Readings:")
    print(abnormal_bp.head())  # Print first few rows for brevity

    print("\nHigh Cholesterol Readings:")
    print(abnormal_chol.head())

    # STEP 3: Save Abnormal Readings to CSV
    os.makedirs("analysis_results", exist_ok=True)  # Create a folder for results
    abnormal_bp.to_csv("analysis_results/abnormal_blood_pressure.csv", index=False)
    abnormal_chol.to_csv("analysis_results/high_cholesterol.csv", index=False)
    print("\nAbnormal readings saved to 'analysis_results' folder.")

    # STEP 4: Visualize Trends (if time-series data is available)
    # The dataset does not appear to include a 'date' column, but we'll keep this as an example:
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        plt.figure(figsize=(10, 6))
        df['systolic_bp'].plot(title='Systolic Blood Pressure Over Time')
        plt.xlabel('Date')
        plt.ylabel('Systolic Blood Pressure')
        plt.savefig("analysis_results/blood_pressure_trend.png")  # Save the plot
        plt.show()

    # STEP 5: Visualize Distributions
    # Example 1: Boxplot of systolic blood pressure
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['systolic_bp'])
    plt.title('Systolic Blood Pressure Distribution')
    plt.savefig("analysis_results/systolic_bp_distribution.png")  # Save the plot
    plt.show()

    # Example 2: Histogram of total cholesterol
    plt.figure(figsize=(10, 6))
    sns.histplot(df['total_cholesterol'], kde=True)
    plt.title('Total Cholesterol Distribution')
    plt.savefig("analysis_results/total_cholesterol_distribution.png")  # Save the plot
    plt.show()

    # STEP 6: Correlation Analysis
    # We’ll consider age, systolic_bp, diastolic_bp, total_cholesterol:
    correlation_columns = ['age', 'systolic_bp', 'diastolic_bp', 'total_cholesterol']
    correlation_matrix = df[correlation_columns].corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Vitals')
    plt.savefig("analysis_results/correlation_matrix.png")  # Save the plot
    plt.show()

    print("\nAnalysis completed. Check the 'analysis_results' folder for outputs.")
else:
    print("Failed to load the dataset.")
