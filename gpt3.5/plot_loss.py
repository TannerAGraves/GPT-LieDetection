import pandas as pd
import matplotlib.pyplot as plt

# Load data from the file
file_path = 'train_full.tsv'
data = pd.read_csv(file_path, sep='\t')

# Plotting the graph
plt.figure(figsize=(10, 6))

# Training loss plot
plt.plot(data['Step'], data['Training loss'], label='Training Loss')

# Validation loss plot
plt.plot(data['Step'], data['Validation loss'], label='Validation Loss')

# Adding labels and title
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Steps')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
