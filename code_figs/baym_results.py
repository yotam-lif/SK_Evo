import os, io
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from scipy.stats import gaussian_kde
import seaborn as sns


def create_dfe_comparison_ridgeplot(ax):
    """
    Creates the DFE comparison ridgeplot using the same FacetGrid-based
    code as in the notebook. The output is saved to an in-memory image and
    then drawn on the provided axis. The facet rows are reversed and no title
    is added.
    """

    # --- Load and process the data ---
    datapath = os.path.join('..', 'data', 'anurag_data', 'Analysis', 'Part_3_TnSeq_analysis',
                            'Processed_data_for_plotting')
    dfe_data_csv = os.path.join(datapath, "dfe_data_pandas.csv")
    dfe_data = pd.read_csv(dfe_data_csv)
    if "Fitness estimate" in dfe_data.columns:
        dfe_data.rename(columns={'Fitness estimate': 'Fitness effect'}, inplace=True)

    # Split into Ara- and Ara+ and then recombine.
    dfe_data_minus = dfe_data[dfe_data['Ara Phenotype'] == 'Ara-']
    dfe_data_plus = dfe_data[dfe_data['Ara Phenotype'] == 'Ara+']
    dfe_data = pd.concat([dfe_data_minus, dfe_data_plus])

    # Rename populations.
    pop_names_old = ["REL606", "REL607", "Ara-1", "Ara-2", "Ara-3",
                     "Ara-4", "Ara-5", "Ara-6", "Ara+1", "Ara+2",
                     "Ara+3", "Ara+4", "Ara+5", "Ara+6"]
    libraries2 = ["Anc", "Anc*", "Ara–1", "Ara–2", "Ara–3",
                  "Ara–4", "Ara–5", "Ara–6", "Ara+1", "Ara+2",
                  "Ara+3", "Ara+4", "Ara+5", "Ara+6"]
    for i, old in enumerate(pop_names_old):
        new = libraries2[i]
        condition = dfe_data['Population'] == old
        dfe_data.loc[condition, 'Population'] = new

    # Exclude unwanted populations.
    dfe_data = dfe_data[~dfe_data['Population'].isin(['Ara–2', 'Ara+4'])]

    # Filter for near-neutral fitness effects.
    dfe_ridge_data = dfe_data[(dfe_data["Fitness effect"] < 0.05) &
                              (dfe_data["Fitness effect"] > -0.1)].copy()

    # --- Reverse the plotting order ---
    # Define the explicit order and then reverse it.
    k_reordered = [0, 2, 4, 5, 6, 7, 1, 8, 9, 10, 12, 13]
    pop_order = [libraries2[k] for k in k_reordered if libraries2[k] not in ['Ara–2', 'Ara+4']]
    pop_order = pop_order[::-1]  # reverse the order

    # Set Population as an ordered categorical variable.
    dfe_ridge_data['Population'] = pd.Categorical(
        dfe_ridge_data['Population'],
        categories=pop_order,
        ordered=True
    )

    # Define the color palette and reverse it.
    ridgeplot_palette = []
    for k in k_reordered:
        if libraries2[k] in ['Ara–2', 'Ara+4']:
            continue
        if k < 2:
            ridgeplot_palette.append('#c0bfbf')
        else:
            ridgeplot_palette.append('#08519c')
    ridgeplot_palette = ridgeplot_palette[::-1]

    # --- Create the FacetGrid (exactly as in the notebook) ---
    g = sns.FacetGrid(dfe_ridge_data, row="Population", hue="Population",
                      aspect=7, height=0.3, palette=ridgeplot_palette, sharex=True)
    sns.set_theme(context="paper", rc={"axes.facecolor": (0, 0, 0, 0)},
                  style='white')
    g.map(sns.kdeplot, "Fitness effect",
          bw_adjust=0.5, clip_on=True,
          fill=True, alpha=1, linewidth=1)
    g.map(sns.kdeplot, "Fitness effect",
          clip_on=False, color="white", lw=0.5, bw_adjust=0.25)

    # Add labels in axes coordinates.
    def label(x, color, label_name):
        ax_temp = plt.gca()
        ax_temp.text(0.03, 0.2, label_name, color='black', size=8,
                     ha="left", va="center", transform=ax_temp.transAxes)

    g.map(label, "Fitness effect")

    # Remove facet titles.
    g.set_titles("")

    # --- Render the FacetGrid to an in-memory image and plot it ---
    buf = io.BytesIO()
    g.fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = mpimg.imread(buf)
    buf.close()

    # Clear the provided axis and display the image.
    ax.clear()
    ax.imshow(img)
    ax.axis('off')


def create_overlapping_dfes(ax):
    """
    Minimal steps to produce the overlapping DFEs figure.
    Based on the code from alex_code/overlapping_dfes.py with modified data path.
    """
    data_path = os.path.join('data', 'alex_code', 'overlapping_dfes_data.csv')
    data = pd.read_csv(data_path)
    ax.plot(data['x'], data['y1'], label='DFE1', color='blue', lw=1.5)
    ax.plot(data['x'], data['y2'], label='DFE2', color='red', lw=1.5)
    ax.legend(frameon=False, fontsize=10)
    ax.set_title("Overlapping DFEs", fontsize=14)
    ax.set_xlabel("X-axis", fontsize=12)
    ax.set_ylabel("Y-axis", fontsize=12)

def create_segben(ax):
    """
    Minimal steps to produce the segben figure.
    Based on the code in alex_code/segben.py with modified data path.
    """
    data_path = os.path.join('data', 'alex_code', 'segben_data.csv')
    data = pd.read_csv(data_path)
    ax.scatter(data['x'], data['y'], color='green', s=25)
    ax.set_title("Segben", fontsize=14)
    ax.set_xlabel("X-axis", fontsize=12)
    ax.set_ylabel("Y-axis", fontsize=12)

def main():
    # Create a figure with 2 rows and 3 columns.
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, wspace=0.3, hspace=0.3)

    # Merge the left-most cells ([0,0] and [1,0]) into one tall axis.
    ax_ridge = fig.add_subplot(gs[:, 0])  # spans both rows in column 0

    # Create the remaining four axes.
    ax_top_middle = fig.add_subplot(gs[0, 1])
    ax_top_right = fig.add_subplot(gs[0, 2])
    ax_bottom_middle = fig.add_subplot(gs[1, 1])
    ax_bottom_right = fig.add_subplot(gs[1, 2])

    # Plot into each axis.
    create_dfe_comparison_ridgeplot(ax_ridge)
    # create_overlapping_dfes(ax_top_middle)
    # create_segben(ax_top_right)
    # # For demonstration, reusing the same functions in the bottom row.
    # create_overlapping_dfes(ax_bottom_middle)
    # create_segben(ax_bottom_right)

    # Add panel labels.
    # Here, we label the merged left axis as "A", then the remaining panels as "B", "C", "D", and "E".
    labels = {
        ax_ridge: "A",
        ax_top_middle: "B",
        ax_top_right: "C",
        ax_bottom_middle: "D",
        ax_bottom_right: "E"
    }
    for ax, label in labels.items():
        ax.text(0.01, 0.95, label, transform=ax.transAxes,
                fontsize=16, fontweight='heavy', va='top', ha='left')

    # Save the figure.
    output_dir = os.path.join('..', 'figs_paper')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "baym_results.svg")
    fig.savefig(output_path, format="svg", bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    main()
