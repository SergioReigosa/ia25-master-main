"""
Visualization utilities for thyroid disease analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_histograms_by_class(df, target_col, columns=None, kde=False):
    """
    Plots univariate histograms for each feature, colored by the target class.
    
    Parameters:
    - df: The dataframe containing features and the target.
    - target_col: The name of the column with the class labels (e.g., 'diagnosis').
    - columns: List of columns to plot. If None, uses all numeric columns.
    """
    
    # 1. Select numeric columns to plot (excluding the target itself)
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in columns:
            columns.remove(target_col)

    # 2. Setup the figure grid
    # We calculate how many rows we need based on the number of columns
    n_cols = 3
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() # Flatten to 1D array for easy iteration

    # 3. Loop through features and plot
    for i, col in enumerate(columns):
        if kde:
            sns.kdeplot(
                data=df, 
                x=col, 
                hue=target_col, 
                common_norm=False, 
                fill=True, 
                alpha=0.5,
                palette='bright',
                ax=axes[i]
            )
        else:
            sns.histplot(
                data=df, 
                x=col, 
                hue=target_col,      # Color by class
                element="step",      # Use 'step' to see overlaps clearly
                stat="count",        # Show raw counts
                common_norm=False,   # Normalize each class independently? False = raw counts
                kde=False,           # Set True if you want a smooth line
                palette='bright',    # High contrast colors
                ax=axes[i]
            )
        axes[i].set_title(f'Distribution of {col} by Class', fontsize=14)
        axes[i].set_xlabel('')

    # 4. Hide empty subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()





# --- Usage ---
# Ensure you provide the correct name of your target column (e.g., 'diagnosis', 'Class')
# plot_histograms_by_class(df, target_col='Class')