# src/visualization.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_pie_chart(
    data,
    title="Pie Chart",
    autopct="%1.1f%%",
    startangle=90,
    figsize=(6, 6)
):
    """
    Plots a pie chart from a pandas Series or DataFrame column.

    Parameters:
    - data: pandas Series (index = labels, values = sizes)
    - title: chart title
    - autopct: percentage format
    - startangle: rotation angle
    - figsize: figure size
    """

    plt.figure(figsize=figsize)
    data.plot(kind="pie", autopct=autopct, startangle=startangle)
    plt.ylabel("")
    plt.title(title)
    plt.tight_layout()
    plt.show()



def plot_bar_chart(
    data,
    y=None,
    title="Bar Chart",
    xlabel=None,
    ylabel=None,
    figsize=(8, 5),
    rotation=0
):
    """
    Plots a bar chart from a pandas Series or DataFrame.

    Parameters:
    - data: pandas Series or DataFrame
    - y: column name (required if data is DataFrame)
    """

    plt.figure(figsize=figsize)

    if isinstance(data, pd.Series):
        data.plot(kind="bar")

    elif isinstance(data, pd.DataFrame):
        if y is None:
            raise ValueError("For DataFrame input, 'y' column must be specified.")
        data.set_index(data.columns[0])[y].plot(kind="bar")

    else:
        raise TypeError("Data must be a pandas Series or DataFrame.")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()



def plot_line_chart(
    data,
    y=None,
    title="Line Chart",
    xlabel=None,
    ylabel=None,
    figsize=(8, 5),
    marker="o"
):
    """
    Plots a line chart from a pandas Series or DataFrame.

    Parameters:
    - data: pandas Series or DataFrame
    - y: column name (required if data is DataFrame)
    """

    plt.figure(figsize=figsize)

    if isinstance(data, pd.Series):
        data.plot(kind="line", marker=marker)

    elif isinstance(data, pd.DataFrame):
        if y is None:
            raise ValueError("For DataFrame input, 'y' column must be specified.")
        data.set_index(data.columns[0])[y].plot(kind="line", marker=marker)

    else:
        raise TypeError("Data must be a pandas Series or DataFrame.")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()
