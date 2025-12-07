"""
Command-line interface providing access to image processing operations such as
prediction, resizing, grayscale conversion, and normalization. The CLI wraps
functions from the mylib.model module and exposes them as terminal commands.
"""

import click
from PIL import Image
from mylib.model import (
    predict_class,
    resize_image,
    convert_to_grayscale,
    normalize_image,
)


@click.group()
def cli():
    """Main CLI group for image processing commands."""


@cli.command()
@click.argument("image_path")
def predict(image_path):
    """Predict the class of an image."""
    image = Image.open(image_path)
    result = predict_class(image)
    click.echo(result)


@cli.command()
@click.argument("image_path")
@click.argument("width", type=int)
@click.argument("height", type=int)
def resize(image_path, width, height):
    """Resize an image to WIDTH and HEIGHT."""
    image = Image.open(image_path)
    resized = resize_image(image, (width, height))
    click.echo(f"{resized.size[0]}x{resized.size[1]}")


@cli.command()
@click.argument("image_path")
def grayscale(image_path):
    """Convert an image to grayscale."""
    image = Image.open(image_path)
    gray = convert_to_grayscale(image)
    click.echo(gray.mode)


@cli.command()
@click.argument("image_path")
def normalize(image_path):
    """Normalize image pixels to the range [0, 1]."""
    image = Image.open(image_path)
    array = normalize_image(image)
    click.echo(f"min={array.min():.2f}, max={array.max():.2f}")


if __name__ == "__main__":
    cli()
