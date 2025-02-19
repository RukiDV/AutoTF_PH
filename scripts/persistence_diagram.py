import matplotlib.pyplot as plt

def plot_persistence_diagram(file_path):
    # read persistence pairs
    persistence_pairs = []
    with open(file_path, 'r') as file:
        for line in file:
            birth, death = map(float, line.strip().split())
            persistence_pairs.append((birth, death))

    plt.figure(figsize=(8, 8))
    for birth, death in persistence_pairs:
        plt.scatter(birth, death, color='blue', alpha=0.7)

    # add diagonal death birth 
    max_value = max([max(birth, death) for birth, death in persistence_pairs] + [1])
    plt.plot([0, max_value], [0, max_value], color='red', linestyle='--', label='Birth = Death')

    plt.title("Persistence Diagram")
    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_persistence_diagram("persistence_pairs.txt")