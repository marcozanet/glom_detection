import inspect
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score 

average_methods = [precision_score, recall_score, f1_score]

def calculate_metric(metric_fn, true_y, pred_y):
    # multi class problems need to have averaging method
    if metric_fn in average_methods:
        return metric_fn(true_y, pred_y, average="macro")
    else:
        return metric_fn(true_y, pred_y)
    
def print_scores(p, r, f1, a, batch_size):
    # just an utility printing function
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")