import click
import pandas as pd
import numpy as np


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def add_features(input_path: str, output_path: str):
    """
    Add new features to the dataset.

    INPUT_PATH: path to cleaned data CSV
    OUTPUT_PATH: path to save dataset with new features
    """

    df = pd.read_csv(input_path)

    df['ratio_day_night_calls'] = round(df['total_day_calls'] / df['total_night_calls'], 2)
    df['ratio_day_night_calls'] = df['ratio_day_night_calls'].replace([np.inf, -np.inf], 0)

    df.to_csv(output_path, index=False)
    click.echo(f"Data with new features saved to {output_path}")


if __name__ == "__main__":
    add_features()
