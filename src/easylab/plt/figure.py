# from typing import cast
# from matplotlib import projections, pyplot as plt
# from matplotlib.axes import Axes
# from matplotlib.figure import Figure

# from ..data import Graph


# class EasylabFigure(Figure):
#     pass


# class EasylabAxes(Axes):
#     name = "easylab_axes"

#     def plot(self, *args, **kwargs):
#         if len(args) == 1 and isinstance(args[0], Graph):
#             graph = args[0]
#             return super().plot(graph.x_arr, graph.y_arr, **kwargs)

#         return super().plot(*args, **kwargs)


# projections.register_projection(EasylabAxes)


# def figure(*args, **kwargs):
#     return plt.figure(*args, FigureClass=EasylabFigure, **kwargs)


# def subplots(*args, **kwargs) -> tuple[EasylabFigure, EasylabAxes]:
#     subplot_kw = {"projection": "easylab_axes", **(kwargs.get("subplot_kw", {}))}
#     del kwargs["subplot_kw"]

#     figure, axes = plt.subplots(
#         *args, FigureClass=EasylabFigure, subplot_kw=subplot_kw, **kwargs
#     )

#     return cast(EasylabFigure, figure), cast(EasylabAxes, axes)
