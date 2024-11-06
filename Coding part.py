import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
csv_file_path = 'gym_members_exercise_tracking.csv'
data = pd.read_csv(csv_file_path)

# Function to display statistical summary
def display_statistics(df):
    print("Statistical Summary (describe):\n", df.describe())
    
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    print("\nCorrelation Matrix:\n", numeric_df.corr())
    
    # Calculate skewness and kurtosis for each numeric column
    print("\nSkewness:\n", numeric_df.skew())
    print("\nKurtosis:\n", numeric_df.kurtosis())

# Histogram for Age Distribution
def plot_age_distribution(df):
    plt.figure(figsize=(8, 6))
    plt.hist(df['Age'], bins=10, color='skyblue', edgecolor='black')
    plt.title("Age Distribution of Gym Members", fontsize=14, fontweight='bold')
    plt.xlabel("Age", fontsize=12, fontweight='bold')
    plt.ylabel("Frequency", fontsize=12, fontweight='bold')
    plt.show()

# Scatter Plot for Calories Burned vs. Session Duration with Gender differentiation
def plot_calories_vs_duration(df):
    plt.figure(figsize=(8, 6))
    colors = {'Male': 'blue', 'Female': 'red'}
    for gender in df['Gender'].unique():
        subset = df[df['Gender'] == gender]
        plt.scatter(subset['Session_Duration (hours)'], subset['Calories_Burned'], 
                    color=colors[gender], label=gender, alpha=0.6)
    
    plt.title("Calories Burned vs. Session Duration", fontsize=14, fontweight='bold')
    plt.xlabel("Session Duration (hours)", fontsize=12, fontweight='bold')
    plt.ylabel("Calories Burned", fontsize=12, fontweight='bold')
    plt.legend(title="Gender", fontsize=10, title_fontsize='12', loc='upper left', frameon=True)
    plt.show()

# Heatmap for Correlation Matrix
def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Exclude non-numeric columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='viridis', fmt=".2f")
    plt.title("Correlation Heatmap of Features", fontsize=14, fontweight='bold')
    plt.show()

# Box Plot for Calories Burned by Workout Type
def plot_calories_by_workout_type(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Workout_Type', y='Calories_Burned', data=df, palette='Set2')
    plt.title("Calories Burned by Workout Type", fontsize=14, fontweight='bold')
    plt.xlabel("Workout Type", fontsize=12, fontweight='bold')
    plt.ylabel("Calories Burned", fontsize=12, fontweight='bold')
    plt.xticks(rotation=45)
    plt.show()

# Execute all steps in sequence
display_statistics(data)
plot_age_distribution(data)
plot_calories_vs_duration(data)
plot_correlation_heatmap(data)
plot_calories_by_workout_type(data)