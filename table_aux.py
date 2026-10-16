# %%
import numpy as np
import pandas as pd


# %%
def pretty_print_matrix(mat : np.ndarray, precision=4, format="e") -> None:
    "Pretty print a matrix with specified precision and format."
    for row in mat:
        print(" ".join(f"{elem: .{precision}{format}}" for elem in row))


if __name__ == "__main__":
    pretty_print_matrix(np.array([[1, 2], [3, 4]]))


# %%
def table_from_dicts(rows: list[dict]) -> pd.DataFrame:
    """Create a pandas DataFrame from a list of dictionaries representing table rows."""
    if len(rows) == 0:
        return pd.DataFrame()
    cols = {key: [] for key in rows[0].keys()}
    for i, row in enumerate(rows):
        if not (row.keys() == cols.keys()):
            raise ValueError(
                f"Row {i} has different keys: {row.keys()}. Expected: {cols.keys()}"
            )
        for key, value in row.items():
            cols[key].append(value)
    return pd.DataFrame(cols)


if __name__ == "__main__":
    print(table_from_dicts([{"a": 1, "b": 2}, {"a": 3, "b": 4}]))

# %%
