training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]

header = ["color", "diameter", "label"]

def class_counts(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        counts[label] = counts.get(label, 0) + 1
    print(f"    class_counts: {counts}")
    return counts

def partition(rows, column, value):
    print(f"partition executed: checking for value {value}")
    true_rows, false_rows = [], []
    for row in rows:
        if match(row, column, value):
            true_rows.append(row)
        else:
            false_rows.append(row)
    print(f"true_rows: {true_rows}")
    print(f"false_rows: {false_rows}")
    return true_rows, false_rows

def match(example, column, value):
    val = example[column]
    return val >= value if isinstance(val, int) or isinstance(val, float) else val == value

def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

def info_gain(left, right, current_uncertainty):
    print(f"info_gain executed: updating current_uncertainty: {current_uncertainty}")
    p = float(len(left)) / (len(left) + len(right))
    print("    current_uncertainty = current_uncertainty - p * gini(left) - (1 - p) * gini(right)")
    print(f"    updated uncertainty = {current_uncertainty} - {p} * {gini(left)} - (1 - {p}) * {gini(right)}")
    current_uncertainty = current_uncertainty - p * gini(left) - (1 - p) * gini(right)
    print(f"    updated uncertainty: {current_uncertainty}")
    return current_uncertainty

def build_tree(rows):
    print("build_tree executed")
    best_gain, best_question = find_best_split(rows)
    if best_gain == 0:
        return {'type': 'leaf', 'predictions': class_counts(rows)}
    col, val = best_question
    true_rows, false_rows = partition(rows, col, val)
    return {
        'type': 'decision',
        'column': col,
        'value': val,
        'true_branch': build_tree(true_rows),
        'false_branch': build_tree(false_rows)
    }

def find_best_split(rows):
    print("find_best_split executed")
    best_gain = 0
    best_question = None
    # current_uncertainty ist AM ANFANG gleich impurity
    current_uncertainty = gini(rows)
    print(f"current_uncertainty: {current_uncertainty}")
    n_features = len(rows[0]) - 1
    for col in range(n_features):
        values = set([row[col] for row in rows]) 
        print(f"col {col}: values: {values}")
        for val in values:
            true_rows, false_rows = partition(rows, col, val)
            if len(true_rows) == 0 or len(false_rows) == 0:
                print("    Partition Ends! len(true_rows) OR len(false_rows) = 0")
                continue
            gain = info_gain(true_rows, false_rows, current_uncertainty)
            if gain > best_gain:
                best_gain, best_question = gain, (col, val)
    print(f"BEST GAIN: {best_gain}, BEST QUESTION (col, val): {best_question}")
    print(f"true_rows: {true_rows}\nfalse_rows: {false_rows}")
    print("\n\n")
    return best_gain, best_question

def print_tree(node, spacing=""):
    print("print_tree executed")
    if node['type'] == 'leaf':
        print(f"{spacing}Predict {node['predictions']}")
        return
    condition = "=="
    if isinstance(node['value'], int) or isinstance(node['value'], float):
        condition = ">="
    print(f"{spacing}Is {header[node['column']]} {condition} {node['value']}?")
    print(f"{spacing}--> True:")
    print_tree(node['true_branch'], spacing + "  ")
    print(f"{spacing}--> False:")
    print_tree(node['false_branch'], spacing + "  ")

def classify(row, node):
    print("classify executed")
    if node['type'] == 'leaf':
        return node['predictions']
    if match(row, node['column'], node['value']):
        return classify(row, node['true_branch'])
    else:
        return classify(row, node['false_branch'])

def print_leaf(counts):
    print("print_leaf executed")
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

my_tree = build_tree(training_data)
print_tree(my_tree)

print("")
testing_data = [
    ["Red", 1],
    ["Red", 1],
    ["Red", 1],
    ["Yellow", 3],  # An unseen feature combination
]


for row in testing_data:
    print(f"Test data: {row}. Predicted: {print_leaf(classify(row, my_tree))}")
