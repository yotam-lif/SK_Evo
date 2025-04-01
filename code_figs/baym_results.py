import os, io
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm

# set font
plt.rcParams['font.family'] = 'sans-serif'


def create_dfe_comparison_ridgeplot(ax_container):
    """
    Creates a ridgeline plot with overlapping KDE curves.
    Each population is plotted in a transparent subplot so that
    only the KDEs overlap (not the white subplot backgrounds).
    Every facet displays an x-axis line (thicker than before). Only the bottom facet shows numeric tick labels,
    while all facets have the population label (e.g., "Anc", "Ara–1", etc.) added inside.
    """
    # --- Data Loading & Processing ---
    datapath = os.path.join('..', 'data', 'anurag_data', 'Analysis',
                            'Part_3_TnSeq_analysis', 'Processed_data_for_plotting')
    dfe_data_csv = os.path.join(datapath, "dfe_data_pandas.csv")
    dfe_data = pd.read_csv(dfe_data_csv)
    if "Fitness estimate" in dfe_data.columns:
        dfe_data.rename(columns={'Fitness estimate': 'Fitness effect'}, inplace=True)

    # Combine Ara- and Ara+ data
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

    # Filter near-neutral fitness effects.
    dfe_ridge_data = dfe_data[(dfe_data["Fitness effect"] < 0.05) &
                              (dfe_data["Fitness effect"] > -0.1)].copy()

    # --- Define Order ---
    k_reordered = [0, 2, 4, 5, 6, 7, 1, 8, 9, 10, 12, 13]
    pop_order = [libraries2[k] for k in k_reordered if libraries2[k] not in ['Ara–2', 'Ara+4']]
    pop_order = pop_order[::-1]  # reverse order

    dfe_ridge_data['Population'] = pd.Categorical(
        dfe_ridge_data['Population'],
        categories=pop_order,
        ordered=True
    )

    # --- Define Colors using 'CMRmap' ---
    cmap = cm.get_cmap("CMRmap")
    color_anc = cmap(0.45)  # use a redder hue for Anc, Anc*
    color_other = cmap(0.1)  # use a bluer hue for others

    # --- Create Nested Grid with Negative Vertical Spacing ---
    fig = ax_container.figure
    container_spec = ax_container.get_subplotspec()
    num_facets = len(pop_order)
    # Use negative hspace to force overlapping
    inner_gs = GridSpecFromSubplotSpec(nrows=num_facets, ncols=1,
                                       subplot_spec=container_spec,
                                       hspace=-0.5)

    # Hide the container axis
    ax_container.set_visible(False)

    # --- Plot Each Facet ---
    for i, pop in enumerate(pop_order):
        ax_facet = fig.add_subplot(inner_gs[i])
        # Make the subplot background transparent
        ax_facet.set_facecolor("none")
        # Set the zorder so that later subplots appear on top
        ax_facet.set_zorder(i)

        pop_data = dfe_ridge_data[dfe_ridge_data["Population"] == pop]

        # Determine fill color based on population
        if pop in ["Anc", "Anc*"]:
            fill_color = color_anc
        else:
            fill_color = color_other

        # Plot the filled KDE curve
        sns.kdeplot(
            data=pop_data, x="Fitness effect",
            bw_adjust=0.5, clip=(-0.1, 0.05),
            fill=True, alpha=1, linewidth=2,
            color=fill_color, ax=ax_facet
        )
        # Plot the white outline for clarity
        sns.kdeplot(
            data=pop_data, x="Fitness effect",
            bw_adjust=0.25, clip=(-0.1, 0.05),
            color="white", lw=1.25, ax=ax_facet
        )

        # Remove y-axis ticks and labels
        ax_facet.set_ylabel("")
        ax_facet.set_yticks([])
        for spine in ax_facet.spines.values():
            spine.set_visible(False)

        # Ensure the bottom spine (x-axis line) is visible and thicker
        ax_facet.spines['bottom'].set_visible(True)
        ax_facet.spines['bottom'].set_linewidth(2)

        # Make the x-axis visible on every facet.
        if i < num_facets - 1:
            # For non-bottom facets: show the x-axis line without tick marks or labels.
            ax_facet.tick_params(axis='x', which='both', length=0, labelbottom=False)
            ax_facet.set_xlabel("")
        else:
            # For the bottom facet: show tick labels (numeric values) without tick marks.
            ax_facet.set_xlim(-0.1, 0.05)
            ax_facet.set_xticks([-0.1, -0.05, 0, 0.05])
            ax_facet.tick_params(axis='x', which='both', length=0, labelbottom=True)
            ax_facet.set_xlabel("Fitness effect", fontsize=16)
            for tick in ax_facet.get_xticklabels():
                tick.set_fontsize(16)

        # Add the population label inside each facet.
        ax_facet.text(0.03, 0.2, pop, color='black', size=18,
                      ha="left", va="center", transform=ax_facet.transAxes)


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
    ax.set_title("B", loc="left", fontsize=16, fontweight="heavy")
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
    ax.set_title("C", loc="left", fontsize=16, fontweight="heavy")
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

    # Plot the ridgeline (subfigure A)
    create_dfe_comparison_ridgeplot(ax_ridge)

    # Plot other subfigures with their titles (panel labels "B" and "C" will be added here)
    # create_overlapping_dfes(ax_top_middle)
    # create_segben(ax_top_right)
    # create_overlapping_dfes(ax_bottom_middle)
    # create_segben(ax_bottom_right)

    # Panel labels are added here as before.
    labels = {
        ax_ridge: "A",
        ax_top_middle: "B",
        ax_top_right: "C",
        ax_bottom_middle: "D",
        ax_bottom_right: "E"
    }
    for ax, label in labels.items():
        ax.text(-0.01, 1.1, label, transform=ax.transAxes,
                fontsize=16, fontweight='heavy', va='top', ha='left')
    ax_top_middle.text(-1.4, 1.1, "A", transform=ax_top_middle.transAxes,
                fontsize=16, fontweight='heavy', va='top', ha='left')

    # Save the figure.
    output_dir = os.path.join('..', 'figs_paper')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "baym_results.svg")
    fig.savefig(output_path, format="svg", bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()
