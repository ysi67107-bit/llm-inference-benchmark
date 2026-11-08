
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULT_DIR = Path("results")
FIGURE_DIR = RESULT_DIR / "figures"

SERVER_PARALLEL = [1, 2, 4]
CLIENT_CONCURRENCY = [1, 2, 4, 8]


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    throughput = np.full((3, 4), np.nan)
    latency = np.full((3, 4), np.nan)

    for row_idx, parallel in enumerate(SERVER_PARALLEL):
        files = sorted(
            RESULT_DIR.glob(
                f"matrix_p{parallel}_summary_*.csv"
            )
        )

        if not files:
            raise FileNotFoundError(
                f"No summary file for parallel={parallel}"
            )

        csv_path = files[-1]

        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            concurrency = int(row["concurrency"])

            col_idx = CLIENT_CONCURRENCY.index(concurrency)

            throughput[row_idx, col_idx] = float(
                row["mean_throughput"]
            )

            latency[row_idx, col_idx] = float(
                row["mean_latency"]
            )

    def plot_heatmap(data, title, colorbar_label, filename):
        fig, ax = plt.subplots(figsize=(9, 5))

        image = ax.imshow(
            data,
            cmap="YlGnBu",
            aspect="auto",
        )

        ax.set_xticks(range(len(CLIENT_CONCURRENCY)))
        ax.set_xticklabels(CLIENT_CONCURRENCY)

        ax.set_yticks(range(len(SERVER_PARALLEL)))
        ax.set_yticklabels(SERVER_PARALLEL)

        ax.set_xlabel("Client Concurrency")
        ax.set_ylabel("OLLAMA_NUM_PARALLEL")
        ax.set_title(title)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{data[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if image.norm(data[i, j]) > 0.55
                        else "black"
                    ),
                )

        fig.colorbar(
            image,
            ax=ax,
            label=colorbar_label,
        )

        fig.tight_layout()
        output_path = FIGURE_DIR / filename
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

        print(f"Saved: {output_path}")

    plot_heatmap(
        throughput,
        "Ollama Request Throughput",
        "Requests / second",
        "parallel_throughput_heatmap.png",
    )

    plot_heatmap(
        latency,
        "Ollama Mean Request Latency",
        "Latency (seconds)",
        "parallel_latency_heatmap.png",
    )


if __name__ == "__main__":
    main()
