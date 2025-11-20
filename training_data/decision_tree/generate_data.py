import os
import random

def generate_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    folder = os.path.join(base_dir, "training_data", "decision_tree")
    os.makedirs(folder, exist_ok=True)

    path = os.path.join(folder, "tree_data.txt")

    rows = []
    for _ in range(15):
        weather = random.choice(["sunny", "rain"])
        workload = random.choice(["light", "heavy"])
        candy = random.choice(["yes", "no"])

        # Simple rule-based tendency:
        # sunny + light + candy → happy
        # otherwise random so the tree is non-trivial
        if weather == "sunny" and workload == "light" and candy == "yes":
            mood = "happy"
        else:
            mood = random.choice(["happy", "sad"])

        rows.append(f"{weather},{workload},{candy},{mood}")

    with open(path, "w") as f:
        f.write("\n".join(rows))

    print("Generated training file:", path)


if __name__ == "__main__":
    generate_data()
