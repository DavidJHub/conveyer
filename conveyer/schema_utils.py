"""Typed-arrow helpers shared by the module output schemas.

Every output table in this project is written with an explicit
:mod:`pyarrow` schema, built column-by-column so nullable ints, floats-with-
null and ``list<string>`` columns round-trip through parquet without pandas
dtype surprises.
"""

from __future__ import annotations

import os
from typing import List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _null_if_nan(v):
    # rows that passed through pandas (df.to_dict) carry NaN where None was
    # meant — arrow refuses NaN for int/bool columns, so normalise here
    if isinstance(v, float) and pd.isna(v):
        return None
    return v


def to_arrow_table(rows: List[dict], schema: pa.Schema) -> pa.Table:
    cols = {}
    for field in schema:
        vals = [_null_if_nan(r.get(field.name)) for r in rows]
        cols[field.name] = pa.array(vals, type=field.type)
    return pa.table(cols, schema=schema)


def to_dataframe(rows: List[dict], schema: pa.Schema) -> pd.DataFrame:
    return to_arrow_table(rows, schema).to_pandas()


def write_parquet(rows: List[dict], schema: pa.Schema, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    pq.write_table(to_arrow_table(rows, schema), path)
    return path
