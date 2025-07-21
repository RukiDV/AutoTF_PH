import pandas as pd
import matplotlib.pyplot as plt

# Load the metrics CSV
df = pd.read_csv('metrics.csv')

# Compute persistence
df['persistence'] = df['death'] - df['birth']

# Scatter plot: J_arc vs persistence
plt.figure()
plt.scatter(df['persistence'], df['J_arc'])
plt.xlabel('Persistence')
plt.ylabel('J_arc')
plt.title('J_arc vs Persistence')
plt.tight_layout()

# Scatter plot: J_box vs persistence
plt.figure()
plt.scatter(df['persistence'], df['J_box'])
plt.xlabel('Persistence')
plt.ylabel('J_box')
plt.title('J_box vs Persistence')
plt.tight_layout()

# Scatter plot: Precision vs persistence
plt.figure()
plt.scatter(df['persistence'], df['precision'])
plt.xlabel('Persistence')
plt.ylabel('Precision')
plt.title('Precision vs Persistence')
plt.tight_layout()

# Scatter plot: Recall vs persistence
plt.figure()
plt.scatter(df['persistence'], df['recall'])
plt.xlabel('Persistence')
plt.ylabel('Recall')
plt.title('Recall vs Persistence')
plt.tight_layout()

plt.show()
