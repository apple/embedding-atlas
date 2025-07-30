# Jupyter Widget

The Python package also provides a Jupyter widget to use Embedding Atlas in your notebooks.

## Installation

```bash
pip install embedding-atlas
```

## Usage

```python
from embedding_atlas.widget import EmbeddingAtlasWidget

# Create an Embedding Atlas widget without projection
# This widget will show table and charts only, not the embedding view.
EmbeddingAtlasWidget(df)

# Compute text embedding and projection of the embedding
from embedding_atlas.projection import compute_text_projection

compute_text_projection(df, text="description",
    x="projection_x", y="projection_y", neighbors="neighbors"
)

# Create an Embedding Atlas widget with the pre-computed projection
widget = EmbeddingAtlasWidget(df, text="description",
    x="projection_x", y="projection_y", neighbors="neighbors"
)

# Display the widget
widget
```

The widget embeds the Embedding Atlas UI into your notebook. You can make selections in the widget, and then use:

```python
df = widget.selection()
```

to get the selection back as a data frame.
