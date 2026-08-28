import copy

import earthkit.plots
import earthkit.plots.utils.time_utils
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def quickplot(
    da: xr.DataArray,
    /,
    chart=None,
    *,
    method: str = "pcolormesh",
    vrange: None | tuple[float, float] = None,
    error: bool = False,
    title: str = "{default_title}",
    time: None | str = None,
    cr: None | float = None,
    **kwargs,
) -> None:
    if "x" not in kwargs:
        for x in ["lon", "longitude"]:
            if x in da.dims:
                kwargs["x"] = x
                break
    if "y" not in kwargs:
        for y in ["lat", "latitude"]:
            if y in da.dims:
                kwargs["y"] = y
                break
    if "x" in kwargs and "y" in kwargs:
        da = da.isel(
            {d: slice(None) if d in (kwargs["x"], kwargs["y"]) else 0 for d in da.dims}
        )

    da_min = np.nanmin(da)
    da_max = np.nanmax(da)

    if vrange is None:
        vmin = da_min
        vmax = da_max

        if error:
            vmax = max(abs(vmin), abs(vmax))
            vmin = -vmax
    else:
        vmin, vmax = vrange

    # compute the default style that earthkit.plots would apply
    source = earthkit.plots.sources.get_source(da)
    style = copy.deepcopy(
        earthkit.plots.styles.auto.guess_style(
            source,
            units=getattr(da, "units", "1"),
        )
    )
    style._levels = earthkit.plots.styles.levels.Levels(np.linspace(vmin, vmax, 22))
    style._legend_kwargs["ticks"] = np.linspace(vmin, vmax, 5)

    if error:
        style._colors = "coolwarm"

    extend_left = da_min < vmin
    extend_right = da_max > vmax

    extend = {
        (False, False): "neither",
        (True, False): "min",
        (False, True): "max",
        (True, True): "both",
    }[(extend_left, extend_right)]
    style._kwargs["extend"] = extend
    style._legend_kwargs["extend"] = extend

    # extract datetime and provide it for plotting labels
    time = None if time is None else source.metadata(time)
    time = source.metadata("time_counter") if time is None else time
    time = source.metadata("valid_time") if time is None else time
    time = source.metadata("time") if time is None else time

    class DataArray(da.__class__):
        __slots__ = ()

        def datetime(self):
            datetime = (
                None
                if time is None
                else earthkit.plots.utils.time_utils.to_pydatetime(time)
            )
            return {"base_time": datetime, "valid_time": datetime}

    data = da.copy(deep=False)
    data.__class__ = DataArray

    create_chart = chart is None
    chart = earthkit.plots.Map() if chart is None else chart

    chart.ax.fill_between(
        [0, 1],
        [1, 1],
        hatch="X",
        edgecolor="magenta" if error else "white",
        facecolor="lavenderblush" if error else "lightgrey",
        transform=chart.ax.transAxes,
        zorder=-999,
    )

    getattr(chart, method)(data, style=style, **kwargs)

    chart.coastlines()
    chart.gridlines()
    chart.legend()
    chart.title(title.format(default_title=chart._default_title_template))

    if cr is not None:
        t = chart.ax.text(
            0.95,
            0.9,
            rf"$\times$ {np.round(cr, 2)}",
            ha="right",
            va="top",
            transform=chart.ax.transAxes,
        )
        t.set_bbox(dict(facecolor="white", alpha=0.75, edgecolor="black"))

    if create_chart:
        chart.show()

        plt.clf()
        plt.close("all")
