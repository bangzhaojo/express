import pandas as pd
import ast
import argparse
import warnings

warnings.filterwarnings("ignore")

def safe_literal_eval(value):
    try:
        if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
            if "'" in value or '"' in value:
                return ast.literal_eval(value)
            else:
                return [word.strip() for word in value.strip("[]").split(",") if word.strip()]
        elif isinstance(value, str) and value.startswith("[") and value.endswith("]") and "," not in value:
            return [value.strip("[]").strip()]
        elif isinstance(value, list):
            return value
        elif isinstance(value, str) and '[' not in value and value.lower() != 'invalid output':
            return [value.strip()]
        else:
            return "INVALID OUTPUT"
    except (ValueError, SyntaxError):
        return "INVALID OUTPUT"

def calculate_metrics(df, cot=False):
    try:
        df["labels"] = df["labels"].apply(safe_literal_eval)
        df["labels"] = df["labels"].apply(lambda lst: [word.lower() for word in lst] if isinstance(lst, list) else lst)
    except Exception as e:
        print(f"Error in labels conversion: {e}")

    df["output_formatted"] = df["output"].apply(safe_literal_eval)

    df["output_formatted"] = df["output_formatted"].apply(lambda lst: [word.lower() for word in lst] if isinstance(lst, list) else lst)

    df["output_formatted"] = df.apply(
        lambda row: "INVALID OUTPUT"
        if isinstance(row["labels"], list) and isinstance(row["output_formatted"], list) and len(row["labels"]) != len(row["output_formatted"])
        else row["output_formatted"],
        axis=1
    )

    return df

def main():
    parser = argparse.ArgumentParser(description="Clean and validate masked prediction outputs.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--output", help="Optional path to save cleaned CSV. Defaults to input path.")
    args = parser.parse_args()

    file_path = args.input
    out_path = args.output if args.output else args.input

    df = pd.read_csv(file_path)
    df = calculate_metrics(df, cot=args.cot)

    try:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    except Exception:
        pass

    invalid_count = df["output_formatted"].apply(lambda x: x == "INVALID OUTPUT").sum()
    print(f"Valid rows: {len(df) - invalid_count} / {len(df)}")

    df.to_csv(out_path, index=False)
    print(f"Cleaned file saved to: {out_path}")

if __name__ == "__main__":
    main()
