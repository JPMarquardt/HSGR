import os
import matplotlib.pyplot as plt

def find_cv_values_files(root_dir):
    cv_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file == 'cv-values.txt':
                cv_files.append(os.path.join(subdir, file))
    return cv_files

def plot_cv_values(files):
    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()
            values = [[float(x) for x in line.strip().split(' ')] for line in lines]
            values = list(zip(*values))
            plt.plot(values[1], values[2], label=file)

    plt.xlabel('Index')
    plt.ylabel('CV Value')
    plt.title('CV Values from All Files')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    root_dir = './simulations/tests'
    cv_files = find_cv_values_files(root_dir)
    if cv_files:
        plot_cv_values(cv_files)
    else:
        print("No cv-values.txt files found.")