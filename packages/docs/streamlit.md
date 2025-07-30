# Streamlit Component

The Python package also provides a Streamlit component to use Embedding Atlas in your Streamlit apps.

## Installation

```bash
pip install embedding-atlas
```

## Usage

```python
from embedding_atlas.streamlit import embedding_atlas
from embedding_atlas.projection import compute_text_projection

# Compute text embedding and projection of the embedding
compute_text_projection(df, text="description",
    x="projection_x", y="projection_y", neighbors="neighbors"
)

# Create an Embedding Atlas component for a given data frame
value = embedding_atlas(
    df, text="description",
    x="projection_x", y="projection_y", neighbors="neighbors",
    show_table=True
)
```

The returned value is a `dict` with a `predicate` string.
The `predicate` is a SQL expression for the current selection in the component.
You may use DuckDB to query the data frame with the predicate:

```python
import duckdb

predicate = value.get("predicate")
if predicate is not None:
    # Query the data frame with the SQL predicate
    selection = duckdb.query_df(
        df, "dataframe", "SELECT * FROM dataframe WHERE " + predicate
    )
    # Show the selection
    st.dataframe(selection)
```

Note that it's also possible to use the component without projection:

```python
value = embedding_atlas(df)
```

Without `x` and `y` the widget will fall back to a table and charts only mode.
