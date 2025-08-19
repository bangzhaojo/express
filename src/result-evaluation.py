import pandas as pd
import numpy as np
import ast
import argparse
from sklearn.metrics import f1_score
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def calculate_correct_lexicon(row):
    list1, list2 = row["labels"], row["output_formatted"]
    return sum(1 for w1, w2 in zip(list1, list2) if w1 == w2)

def calculate_correct_vector(row, breakdowns):
    def get_vector(word):
        if word in breakdowns['word'].values:
            return breakdowns.loc[breakdowns['word'] == word].iloc[0, 1:].values.astype(int)
        else:
            return np.zeros(10, dtype=int)

    label_vectors = np.array([get_vector(word) for word in row["labels"]])
    output_vectors = np.array([get_vector(word) for word in row["output_formatted"]])

    label_vectors = label_vectors.astype(int)
    output_vectors = output_vectors.astype(int)

    total_correct = sum(np.array_equal(vec1, vec2) for vec1, vec2 in zip(label_vectors, output_vectors))

    vector_f1_scores = [
        f1_score(vec1, vec2, average="binary", zero_division=0)
        for vec1, vec2 in zip(label_vectors, output_vectors)
    ]
    vector_accuracies = [
        np.mean(vec1 == vec2) for vec1, vec2 in zip(label_vectors, output_vectors)
    ]

    return total_correct, vector_f1_scores, vector_accuracies

def calculate_metrics(df, breakdowns):
    total_mask_sum = df["number_of_labels"].sum()
    total_invalid = (df['output_formatted'] == 'INVALID OUTPUT').sum()
    print("VRate =", (len(df) - total_invalid) / len(df))

    valid_df = df[df["output_formatted"] != 'INVALID OUTPUT'].copy()
    valid_df["output_formatted"] = valid_df["output_formatted"].apply(ast.literal_eval)
    valid_df["labels"] = valid_df["labels"].apply(ast.literal_eval)

    tqdm.pandas(desc="Lexicon Accuracy")
    valid_df["total_correct_lexicon"] = valid_df.progress_apply(
        lambda row: calculate_correct_lexicon(row), axis=1
    )

    tqdm.pandas(desc="Vector Accuracy")
    valid_df[["total_correct_vector", "f1_scores", "vector_accuracies"]] = valid_df.progress_apply(
        lambda row: pd.Series(calculate_correct_vector(row, breakdowns)), axis=1
    )

    all_f1_scores = [f1 for f1_list in valid_df["f1_scores"] for f1 in f1_list]
    all_vector_accuracies = [acc for acc_list in valid_df["vector_accuracies"] for acc in acc_list]

    total_correct_lexicon_sum = valid_df["total_correct_lexicon"].sum()
    total_correct_vector_sum = valid_df["total_correct_vector"].sum()

    average_f1_vector = np.mean(all_f1_scores) if all_f1_scores else 0.0
    average_vector_accuracy = np.mean(all_vector_accuracies) if all_vector_accuracies else 0.0

    accuracy_lexicon = total_correct_lexicon_sum / total_mask_sum if total_mask_sum > 0 else 0
    accuracy_vector = total_correct_vector_sum / total_mask_sum if total_mask_sum > 0 else 0

    print("AccL =", accuracy_lexicon)
    print("AccV =", accuracy_vector)
    print("F1V =", average_f1_vector)
    print("AccV-2 =", average_vector_accuracy)

def evaluate_model(result_path, breakdowns_path):
    print("Evaluating:", result_path)
    result = pd.read_csv(result_path)
    breakdowns = pd.read_csv(breakdowns_path)

    print("Total valid rows =", len(result))
    calculate_metrics(result, breakdowns)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model emotion prediction accuracy.")
    parser.add_argument("--result_path", required=True, help="CSV file of model predictions")
    parser.add_argument("--breakdowns_path", required=True, help="CSV of emotion vector breakdowns")

    args = parser.parse_args()

    evaluate_model(
        result_path=args.result_path,
        breakdowns_path=args.breakdowns_path
    )
