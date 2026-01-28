from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import numpy as np
import os

def tree_to_c_code(tree, feature_names, function_name="decision_tree", file=None):
    """
    Generates C code for a decision tree from a scikit-learn RandomForestClassifier.

    Args:
        tree: The decision tree object from a RandomForestClassifier.
        feature_names: A list of feature names.
        function_name: The name of the C function to generate.
        file: File object to write the code to. If None, prints to stdout.
    """
    def write_line(line, indent=0):
        indent_str = "    " * indent
        if file:
            file.write(f"{indent_str}{line}\n")
        else:
            print(f"{indent_str}{line}")

    def recurse(node, depth):
        if tree.feature[node] != -2:  # Not a leaf node
            name = feature_names[tree.feature[node]]
            threshold = tree.threshold[node]
            write_line(f"if (X[{tree.feature[node]}] <= {threshold:.6f}) {{", depth)
            recurse(tree.children_left[node], depth + 1)
            write_line("} else {", depth)
            recurse(tree.children_right[node], depth + 1)
            write_line("}", depth)
        else:  # Leaf node
            class_id = np.argmax(tree.value[node])
            write_line(f"return {class_id};", depth)

    # Write complete C program with includes and main function
    write_line("#include <stdio.h>")
    write_line("")
    write_line(f"int {function_name}(float X[]) {{")
    recurse(0, 1)
    write_line("}")
    write_line("")
    # Add a main function to test the tree
    write_line("int main() {")
    write_line("    float test_sample[] = {5.1f, 3.5f, 1.4f, 0.2f};  // Example iris sample", 1)
    write_line("    int result = decision_tree(test_sample);", 1)
    write_line('    printf("Predicted class: %d\\n", result);', 1)
    write_line("    return 0;", 1)
    write_line("}")

# Load dataset and train Random Forest
iris = load_iris()
X, y = iris.data, iris.target
rf = RandomForestClassifier(n_estimators=1, random_state=42)  # One tree for simplicity
rf.fit(X, y)
tree = rf.estimators_[0].tree_

# Feature names
feature_names = iris.feature_names

# Create codegen directory if it doesn't exist
codegen_dir = os.path.join("rfc", "plugin-c", "codegen")
os.makedirs(codegen_dir, exist_ok=True)

# Generate C code and write to file
output_file = os.path.join(codegen_dir, "decision_tree.c")
with open(output_file, "w") as f:
    tree_to_c_code(tree, feature_names, file=f)

print(f"C code has been written to {output_file}")

