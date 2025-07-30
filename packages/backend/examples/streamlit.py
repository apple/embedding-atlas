import duckdb
import pandas as pd
import streamlit as st
from datasets import load_dataset
from embedding_atlas.projection import compute_text_projection
from embedding_atlas.streamlit import embedding_atlas


@st.cache_data
def load_data():
    ds = load_dataset("james-burton/wine_reviews", split="validation")
    return pd.DataFrame(ds)


def main():
    # Embedding Atlas looks better in wide mode
    st.set_page_config(layout="wide")

    st.title("Embedding Atlas + Streamlit")

    # Load some data
    st.write("Load an example dataset")
    df = load_data()

    # Compute text embedding and projection of the embedding
    compute_text_projection(
        df,
        text="description",
        x="projection_x",
        y="projection_y",
        neighbors="neighbors",
    )

    # Create the Embedding Atlas widget in Streamlit
    value = embedding_atlas(
        df,
        text="description",
        x="projection_x",
        y="projection_y",
        neighbors="neighbors",
        show_table=True,
    )

    # Show selected rows in a Streamlit data frame
    st.write("Selected rows:")
    if value is not None and value.get("predicate") is not None:
        subset = duckdb.query_df(
            df, "dataframe", "SELECT * FROM dataframe WHERE " + value.get("predicate")
        )
        st.dataframe(subset)
    else:
        st.write("No selection")


if __name__ == "__main__":
    main()
