import matplotlib.pyplot as plt
import numpy as np


def washing_up_payoff(p, q):
    """p is the probability of one child doing the laundry and q is the probability of
    the other child doing the laundry"""
    return p * q + 2 * p * (1 - q) + 2 * (1 - p) * q + (1 - p) * (1 - q)


def create_contour_plot():
    x = np.arange(0, 1, 1e-4)
    y = np.arange(0, 1, 1e-4)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = washing_up_payoff(xx, yy)

    fig, ax = plt.subplots()
    cs = ax.contourf(x, y, z, cmap="gray")
    ax.scatter([0, 1], [1, 0], c="green", marker="o", label="Unconstrained Optimum", s=100, zorder=4)
    ax.scatter([0.5], [0.5], c="yellow", marker="x", label="Symmetric Optimum", s=100, linewidths=3, zorder=3)
    ax.plot([0, 1], [0, 1], c="purple", linestyle="-", label="Symmetric Strategy Profiles", linewidth=10, zorder=2)
    ax.plot([0, 1], [0.5, 0.5], c="blue", linestyle="--", label="Unilateral Deviations from Symmetric Optimum",
            linewidth=10, zorder=1)
    ax.plot([0.5, 0.5], [0, 1], c="blue", linestyle="--", label="Unilateral Deviations from Symmetric Optimum",
            linewidth=10, zorder=1)
    ax.set_xlabel("Rob's Probability of Laundry")
    ax.set_ylabel("Bot's Probability of Laundry")

    # remove duplicate legend items, h/t https://stackoverflow.com/a/13589144/3025865
    handles, labels = ax.get_legend_handles_labels()
    first_index = labels.index("Unconstrained Optimum")
    second_index = labels.index("Symmetric Optimum")
    third_index = labels.index("Symmetric Strategy Profiles")
    fourth_index = labels.index("Unilateral Deviations from Symmetric Optimum")
    reordered_handles = [handles[first_index], handles[second_index], handles[third_index], handles[fourth_index]]
    reordered_labels = [labels[first_index], labels[second_index], labels[third_index], labels[fourth_index]]
    box = ax.get_position()
    # to place legend below plot
    ax.set_position([box.x0, box.y0 + 0.25 * box.height, box.width, 0.75 * box.height])
    ax.legend(reordered_handles, reordered_labels, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    # # to place legend right of plot
    # ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig.savefig("contour_fig.png")


if __name__ == "__main__":
    create_contour_plot()
