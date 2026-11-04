import pandera.polars as pa


def _template(df):
    """Template
    """

    # Validation
    pa.DataFrameSchema({
    }).validate(df)
    
    # Input

    # Constant
    # n_rows = df.shape[0]

    # Calculation

    # Output
    return (
        df
        .with_columns(
        )
    )

def _check_inputs(df, inputs):
    """Check the inputs for the required columns
    """

    for i in inputs:
        if i not in df.columns:
            print("Missing column", i)
            return i
    
    return None

def _drop_outputs(df, outputs):
    """Drop the outputs to be replaced with new calculated values
    """

    selected = [item for item in outputs if item in df.columns]

    return df.drop(selected)
def bar_layers(df, prefix="dia_vert"):
    """The diameters of every layer of bars present, dia_vert0, dia_vert1, ..., with empty layers as 0 [mm]
    """

    layers = []
    while f"{prefix}{len(layers)}" in df.columns:
        layers.append(df[f"{prefix}{len(layers)}"].fill_null(0).to_numpy())

    return layers
