import ipywidgets as widgets
from ..data import Data, DataLike
import matplotlib.pyplot as plt
from . import Graph, GraphInput


class DataInspector(widgets.VBox):
    def __init__(self, data: DataLike):
        data = Data.interpret(data)

        vars = data.get_vars()
        graph_inputs: list[widgets.HBox] = []
        graphs: list[Graph] = []

        x_query = vars[0].label.ascii

        x_query_text = widgets.Text(
            value=x_query, description="x-axis:", continuous_update=False
        )

        def on_x_query_change(change):
            nonlocal x_query, graphs
            x_query = change.new

            for graph in graphs:
                graph.x_query = x_query

            update_ui()

        x_query_text.observe(on_x_query_change, names="value")  # type: ignore

        add_graph_button = widgets.Button(description="Add graph")

        def on_add_graph_button_click(b):
            graph = Graph(data, x_query, "")

            def on_remove():
                index = graphs.index(graph)
                graph_inputs[index].close()
                graph_inputs.pop(index)
                graphs.pop(index)
                print("removed", index)
                update_ui()

            input = GraphInput(graph, on_update=update_ui, on_remove=on_remove)

            graphs.append(graph)
            graph_inputs.append(input)
            update_ui()

        add_graph_button.on_click(on_add_graph_button_click)

        # def create_plot_var(input_str: str, label: Any):
        #     expr = Expr.parse(
        #         input_str, symbols=self.get_vars()
        #     )  # TODO: Better type inference
        #     return Computed(label, expr)

        plot_output = widgets.Output()
        table_output = widgets.Output()

        def show_exception(e):
            with plot_output:
                _, ax = plt.subplots(figsize=(8, 6))
                ax.text(
                    0.5,
                    0.5,
                    str(e),
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="red",
                    fontdict={"size": 8},
                )
                plt.show()

        # fig, ax = plt.subplots()
        def update_plot():
            table_output.clear_output(True)
            plot_output.clear_output(True)

            with plot_output:
                _, ax = plt.subplots(figsize=(8, 6))
                try:
                    for graph in graphs:
                        graph.plot(ax=ax)

                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    show_exception(e)

        def update_ui():
            update_plot()

            self.children = [
                x_query_text,
                *graph_inputs,
                add_graph_button,
                plot_output,
                # widgets.HBox([plot_output, table_output])
            ]

        super().__init__()

        # Create first graph
        on_add_graph_button_click(None)
