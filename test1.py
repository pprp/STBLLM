import matplotlib.pyplot as plt

# Extract data into lists
layers = []
times = []
errors = []

name2error = {}
name2time = {}

# Assuming the data is stored in a file named 'data.txt'
with open('logs/test.log', 'r') as file:
    for line in file:
        parts = line.split()
        layer_id = parts[0]
        name = parts[1]
        time = float(parts[3])
        error = float(parts[5])
        
        if name not in name2error:
            name2error[name] = error 
            name2time[name] = time 
        else: 
            name2error[name] = (name2error[name] + error) / 2
            name2time[name] = (name2time[name] + time) / 2

# Plot bar chart for name2error 
plt.figure(figsize=(10, 6))
plt.bar(name2error.keys(), name2error.values())
plt.xlabel("Layer")
plt.ylabel("Error")
plt.title("Error per layer")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("logs/error.png")
plt.close()

# Plot bar chart for name2time
plt.figure(figsize=(10, 6))
plt.bar(name2time.keys(), name2time.values())
plt.xlabel("Layer")
plt.ylabel("Time")
plt.title("Time per layer")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("logs/time.png")
plt.close()