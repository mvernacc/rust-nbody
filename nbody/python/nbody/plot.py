import click
import pandas as pd
from matplotlib import pyplot as plt

NUM_COLUMNS_PER_BODY = 9


@click.command()
@click.argument("result_file", type=click.File("r"))
def plot_nbody_result(result_file):
    df = pd.read_csv(result_file)
    n_bodies = (len(df.columns) - 1) // NUM_COLUMNS_PER_BODY
    body_names = [
        df.columns[1 + NUM_COLUMNS_PER_BODY * i].split(" position")[0]
        for i in range(n_bodies)
    ]
    print(f"Loaded trajectory data for the following bodies: {', '.join(body_names)}")

    # Plot position 0, position 1
    plt.figure()
    for i in range(n_bodies):
        name = body_names[i]
        plt.plot(
            df[f"{name} position 0 [m]"],
            df[f"{name} position 1 [m]"],
            label=name,
        )
    plt.xlabel("Position 0 [m]")
    plt.ylabel("Position 1 [m]")
    plt.axis("equal")
    plt.legend()

    # Plot time series
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 8))
    ax_r, ax_v, ax_a = axes
    LINESTYLES_BY_DIMENSION = ["-", "--", ":"]
    for i in range(n_bodies):
        name = body_names[i]
        color = f"C{i % 10:d}"
        for k in range(3):
            plt_kwargs = dict(
                label=f"{name} {k}",
                color=color,
                linestyle=LINESTYLES_BY_DIMENSION[k],
            )
            ax_r.plot(
                df["Time [s]"],
                df[f"{name} position {k} [m]"],
                **plt_kwargs,
            )
            ax_v.plot(
                df["Time [s]"],
                df[f"{name} velocity {k} [m s^-1]"],
                **plt_kwargs,
            )
            ax_a.plot(
                df["Time [s]"],
                df[f"{name} acceleration {k} [m s^-2]"],
                **plt_kwargs,
            )
    ax_r.set_ylabel("Position [m]")
    ax_v.set_ylabel("Velocity [m s$^{-1}$]")
    ax_a.set_ylabel("Acceleration [m s$^{-2}$]")
    ax_a.set_xlabel("Time [s]")
    ax_r.legend()
    ax_v.legend()
    ax_a.legend()

    plt.show()
