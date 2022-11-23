from __future__ import annotations
from typing import Any

from matplotlib import pyplot as plt
from ..data import Data, DataLike, VarQuery
import ipywidgets as widgets


class Graph:
    def __init__(
        self,
        data: DataLike,
        x_query: VarQuery,
        y_query: VarQuery,
        label: str | None = None,
        color: Any = None,
    ) -> None:
        self.data = Data.interpret(data)
        self.x_query = x_query
        self.y_query = y_query
        self.label = label
        self.color = color

    def get_x_var(self):
        return self.data.get_var(self.x_query)

    def get_y_var(self):
        return self.data.get_var(self.y_query)

    def plot(self, ax: plt.Axes):
        x_values = []
        y_values = []
        for record in self.data:
            if self.x_query in record and self.y_query in record:
                x_values.append(record[self.x_query])
                y_values.append(record[self.y_query])

        ax.plot(
            x_values,
            y_values,
            label=self.label,
            color=self.color,
        )


class GraphInput(widgets.HBox):
    def __init__(
        self, graph: Graph, *, on_update=lambda: None, on_remove=lambda: None
    ) -> None:
        label_text = widgets.Text(
            placeholder="label",
            value=graph.label or "",
            continuous_update=False,
            layout={"width": "8rem"},
        )

        def on_label_change(change):
            graph.label = change.new
            on_update()

        label_text.observe(on_label_change, names="value")  # type: ignore

        y_query_text = widgets.Text(
            placeholder="value",
            value=str(graph.y_query),
            continuous_update=False,
            layout={"width": "8rem"},
        )

        def on_y_query_change(change):
            graph.y_query = change.new
            on_update()

        y_query_text.observe(on_y_query_change, names="value")  # type: ignore

        color_picker = widgets.ColorPicker(
            # concise=True,
            placeholder="color",
            value="black",
            layout={"width": "6rem"},
        )

        def on_color_changed(change):
            graph.color = change.new
            on_update()

        color_picker.observe(on_color_changed, names="value")  # type: ignore

        remove_button = widgets.Button(
            # description="remove",
            button_style="danger",
            icon="trash",
            layout={"width": "2rem"},
        )

        def on_remove_button_click(b):
            on_remove()

        remove_button.on_click(on_remove_button_click)

        super().__init__(
            children=(
                label_text,
                y_query_text,
                color_picker,
                remove_button,
            ),
            layout={
                "border": "solid 2px #eee",
                "margin": "0.5rem",
                "width": "100%",
            },
        )
