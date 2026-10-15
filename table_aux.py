# %%
import pandas as pd


# %%
def print_matrix(mat, precision=4, format="e"):
    for row in mat[:6, :6]:
        print(" ".join(f"{elem: .{precision}{format}}" for elem in row))


# %%
def table_from_dicts(rows: list[dict]) -> pd.DataFrame:
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
