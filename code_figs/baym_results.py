import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, Rectangle

# set font
plt.rcParams['font.family'] = 'sans-serif'
label_fontsize = 16
tick_fontsize = 14
color = sns.color_palette('CMRmap', 5)
EVO_FILL = (color[1][0], color[1][1], color[1][2], 0.75)
ANC_FILL = (0.5, 0.5, 0.5, 0.4)

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

    color_anc = ANC_FILL
    color_other = color[0]

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
            ax_facet.set_xlabel(r'Fitness effect $(\Delta)$', fontsize=16)
            for tick in ax_facet.get_xticklabels():
                tick.set_fontsize(16)

        # Add the population label inside each facet.
        ax_facet.text(0.03, 0.2, pop, color='black', size=18,
                      ha="left", va="center", transform=ax_facet.transAxes)


def create_overlapping_dfes(ax_left, ax_right):
    # Vertical shift for the "evolved" histograms
    z = 30
    lw_main = 1.0

    def draw_custom_segments(ax):
        ax.plot([-0.09, 0.09], [z * 1.1, z * 1.1],
                linestyle="--", color="grey", lw=lw_main)
        segs = [
            ((-0.05, -0.75), (-0.05 * 0.9, z * 1.1)),
            ((0.05, -0.75), (0.05 * 0.9, z * 1.1)),
            ((-0.1, -0.75), (-0.09, z * 1.1)),
            ((0.1, -0.75), (0.09, z * 1.1)),
            ((0, -0.75), (0, z * 1.1))
        ]
        for (x0, y0), (x1, y1) in segs:
            ax.plot([x0, x1], [y0, y1], linestyle="--", color="grey", lw=lw_main)

    # Set file path using the datadir variable
    datadir = os.path.join('..', 'alex_code')
    Rtable = pd.read_csv(os.path.join(datadir, "Rfitted_fil.txt"), sep="\t")
    Ttable = pd.read_csv(os.path.join(datadir, "2Kfitted_fil.txt"), sep="\t")
    Ftable = pd.read_csv(os.path.join(datadir, "15Kfitted_fil.txt"), sep="\t")

    # Remove rows with NA in 'fitted1'
    Rtable = Rtable.dropna(subset=["fitted1"])
    Ttable = Ttable.dropna(subset=["fitted1"])
    Ftable = Ftable.dropna(subset=["fitted1"])

    # Filter beneficial alleles: fitted1 in (0.015, 0.3] and abn > 1
    Rben = Rtable[(Rtable["fitted1"] <= 0.3) & (Rtable["fitted1"] > 0.015) & (Rtable["abn"] > 1)].copy()
    Tben = Ttable[(Ttable["fitted1"] <= 0.3) & (Ttable["fitted1"] > 0.015) & (Ttable["abn"] > 1)].copy()
    Fben = Ftable[(Ftable["fitted1"] <= 0.3) & (Ftable["fitted1"] > 0.015) & (Ftable["abn"] > 1)].copy()

    # Remove duplicate sites
    Rben = Rben.drop_duplicates(subset=["site"])
    Tben = Tben.drop_duplicates(subset=["site"])
    Fben = Fben.drop_duplicates(subset=["site"])

    # Build data frames for allele comparisons
    Rnames = Rben["alle"].values
    Repi = pd.DataFrame(np.nan, index=range(len(Rnames)), columns=["R", "M", "K"])
    for i, allele in enumerate(Rnames):
        r_val = Rtable.loc[Rtable["alle"] == allele, "fitted1"]
        if not r_val.empty:
            Repi.at[i, "R"] = r_val.iloc[0]
        t_val = Ttable.loc[Ttable["alle"] == allele, "fitted1"]
        if not t_val.empty:
            Repi.at[i, "M"] = t_val.iloc[0]
        f_val = Ftable.loc[Ftable["alle"] == allele, "fitted1"]
        if not f_val.empty:
            Repi.at[i, "K"] = f_val.iloc[0]

    Tnames = Tben["alle"].values
    Tepi = pd.DataFrame(np.nan, index=range(len(Tnames)), columns=["R", "M", "K"])
    for i, allele in enumerate(Tnames):
        t_val = Ttable.loc[Ttable["alle"] == allele, "fitted1"]
        if not t_val.empty:
            Tepi.at[i, "M"] = t_val.iloc[0]
        r_val = Rtable.loc[Rtable["alle"] == allele, "fitted1"]
        if not r_val.empty:
            Tepi.at[i, "R"] = r_val.iloc[0]
        f_val = Ftable.loc[Ftable["alle"] == allele, "fitted1"]
        if not f_val.empty:
            Tepi.at[i, "K"] = f_val.iloc[0]

    Fnames = Fben["alle"].values
    Fepi = pd.DataFrame(np.nan, index=range(len(Tnames)), columns=["R", "M", "K"])
    for i, allele in enumerate(Fnames):
        f_val = Ftable.loc[Ftable["alle"] == allele, "fitted1"]
        if not f_val.empty:
            Fepi.at[i, "K"] = f_val.iloc[0]
        r_val = Rtable.loc[Rtable["alle"] == allele, "fitted1"]
        if not r_val.empty:
            Fepi.at[i, "R"] = r_val.iloc[0]
        t_val = Ttable.loc[Ttable["alle"] == allele, "fitted1"]
        if not t_val.empty:
            Fepi.at[i, "M"] = t_val.iloc[0]

    # Combine data sets and keep rows with at least 2 non-NA values
    Repi["back"] = 1
    Tepi["back"] = 2
    Fepi["back"] = 3
    data = pd.concat([Repi, Tepi, Fepi], ignore_index=True)
    data["nas"] = data[["R", "M", "K"]].notna().sum(axis=1)
    x = data[data["nas"] >= 2].copy()

    # Left Panel
    vals_after = np.concatenate([
        x.loc[x["back"] == 1, "M"].dropna().values,
        x.loc[x["back"] == 1, "K"].dropna().values
    ])
    counts, bin_edges = np.histogram(vals_after, bins=30)
    bin_edges = bin_edges * 0.9
    counts_shifted = counts + z

    ax_left.set_xlim(-0.1, 0.1)
    ax_left.set_ylim(0, 500)
    ax_left.set_xlabel(r'Fitness effect $(\Delta)$', fontsize=16)
    ax_left.tick_params(labelsize=14)

    draw_custom_segments(ax_left)
    ax_left.stairs(
        values=counts_shifted,
        edges=bin_edges,
        baseline=0,
        fill=True,
        facecolor=EVO_FILL,
        edgecolor="black",
        lw=1.1,
        label="Evolved"
    )
    rect = Rectangle((-0.11, 0), 0.21, z, facecolor="white", edgecolor="none")
    ax_left.add_patch(rect)
    draw_custom_segments(ax_left)
    vals_anc = x.loc[x["back"] == 1, "R"].dropna().values
    anc_counts, anc_bin_edges = np.histogram(vals_anc, bins=24)
    ax_left.stairs(
        values=anc_counts,
        edges=anc_bin_edges,
        baseline=0,
        fill=True,
        facecolor=ANC_FILL,
        edgecolor="black",
        lw=1.1,
        label="Ancestor"
    )
    ax_left.legend(frameon=False)

    # Right Panel
    vals_after2 = np.concatenate([
        x.loc[x["back"] == 2, "M"].dropna().values,
        x.loc[x["back"] == 3, "K"].dropna().values
    ])
    counts2, bin_edges2 = np.histogram(vals_after2, bins=10)
    bin_edges2 = bin_edges2 * 0.9
    counts2_shifted = counts2 + z

    ax_right.set_xlim(-0.1, 0.1)
    ax_right.set_ylim(0, 500)
    ax_right.set_xlabel(r'Fitness effect $(\Delta)$', fontsize=16)
    ax_right.tick_params(labelsize=14)

    draw_custom_segments(ax_right)
    ax_right.stairs(
        values=counts2_shifted,
        edges=bin_edges2,
        baseline=0,
        fill=True,
        facecolor=EVO_FILL,
        edgecolor="black",
        lw=1.1,
        label="Evolved"
    )
    rect2 = Rectangle((-0.11, 0), 0.21, z, facecolor="white", edgecolor="none")
    ax_right.add_patch(rect2)
    draw_custom_segments(ax_right)
    vals_anc2 = np.unique(np.concatenate([
        x.loc[x["back"].isin([2, 3]), "R"].dropna().values
    ]))
    anc2_counts, anc2_bin_edges = np.histogram(vals_anc2, bins=30)
    ax_right.stairs(
        values=anc2_counts,
        edges=anc2_bin_edges,
        baseline=0,
        fill=True,
        facecolor=ANC_FILL,
        edgecolor="black",
        lw=1.1,
        label="Ancestor"
    )
    ax_right.legend(frameon=False)

    # Adjust spines and tick positions for a cleaner look
    for ax in [ax_left, ax_right]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')


def create_segben(ax_B, ax_C):
    # Read tables from datadir
    datadir = os.path.join('..', 'alex_code')
    Rtable = pd.read_csv(os.path.join(datadir, "Rfitted_fil.txt"), sep="\t")
    Ttable = pd.read_csv(os.path.join(datadir, "2Kfitted_fil.txt"), sep="\t")
    Ftable = pd.read_csv(os.path.join(datadir, "15Kfitted_fil.txt"), sep="\t")

    # Remove rows with NA in 'fitted1'
    Rtable = Rtable.dropna(subset=["fitted1"])
    Ttable = Ttable.dropna(subset=["fitted1"])
    Ftable = Ftable.dropna(subset=["fitted1"])

    # Filter beneficial alleles and remove duplicate sites
    Rben = Rtable[(Rtable["fitted1"] <= 0.3) & (Rtable["fitted1"] > 0.015) & (Rtable["abn"] > 1)].drop_duplicates(subset=["site"]).copy()
    Tben = Ttable[(Ttable["fitted1"] <= 0.3) & (Ttable["fitted1"] > 0.015) & (Ttable["abn"] > 1)].drop_duplicates(subset=["site"]).copy()
    Fben = Ftable[(Ftable["fitted1"] <= 0.3) & (Ftable["fitted1"] > 0.015) & (Ftable["abn"] > 1)].drop_duplicates(subset=["site"]).copy()

    # Create data frame for allele comparisons: Repi for the ancestor
    Rnames = Rben["alle"].values
    Repi = pd.DataFrame(np.nan, index=range(len(Rnames)), columns=["R", "M", "K"])
    for i, allele in enumerate(Rnames):
        r_val = Rtable.loc[Rtable["alle"] == allele, "fitted1"]
        if not r_val.empty:
            Repi.at[i, "R"] = r_val.iloc[0]
        t_val = Ttable.loc[Ttable["alle"] == allele, "fitted1"]
        if not t_val.empty:
            Repi.at[i, "M"] = t_val.iloc[0]
        f_val = Ftable.loc[Ftable["alle"] == allele, "fitted1"]
        if not f_val.empty:
            Repi.at[i, "K"] = f_val.iloc[0]

    # Tepi: top alleles in 2K
    Tnames = Tben["alle"].values
    Tepi = pd.DataFrame(np.nan, index=range(len(Tnames)), columns=["R", "M", "K"])
    for i, allele in enumerate(Tnames):
        t_val = Ttable.loc[Ttable["alle"] == allele, "fitted1"]
        if not t_val.empty:
            Tepi.at[i, "M"] = t_val.iloc[0]
        r_val = Rtable.loc[Rtable["alle"] == allele, "fitted1"]
        if not r_val.empty:
            Tepi.at[i, "R"] = r_val.iloc[0]
        f_val = Ftable.loc[Ftable["alle"] == allele, "fitted1"]
        if not f_val.empty:
            Tepi.at[i, "K"] = f_val.iloc[0]

    # Fepi: top alleles in 15K (using same number of rows as Tepi)
    Fnames = Fben["alle"].values
    Fepi = pd.DataFrame(np.nan, index=range(len(Tnames)), columns=["R", "M", "K"])
    for i, allele in enumerate(Fnames):
        f_val = Ftable.loc[Ftable["alle"] == allele, "fitted1"]
        if not f_val.empty:
            Fepi.at[i, "K"] = f_val.iloc[0]
        r_val = Rtable.loc[Rtable["alle"] == allele, "fitted1"]
        if not r_val.empty:
            Fepi.at[i, "R"] = r_val.iloc[0]
        t_val = Ttable.loc[Ttable["alle"] == allele, "fitted1"]
        if not t_val.empty:
            Fepi.at[i, "M"] = t_val.iloc[0]

    color_0k = (color[0][0], color[0][1], color[0][2], 0.5)
    color_2k = (color[1][0], color[1][1], color[1][2], 0.5)
    color_15k = (color[2][0], color[2][1], color[2][2], 0.5)
    # Panel B: ancestral to 2K
    ax_B.set_xlim(0.8, 2.2)
    ax_B.set_ylim(-0.15, 0.1)
    ax_B.set_ylabel(r'Fitness effect $(\Delta)$', fontsize=16)
    ax_B.tick_params(labelsize=14)
    # Plot ancestral fitness values (Repi) at x = 1
    ax_B.plot(np.repeat(1, len(Repi)), Repi["R"], 'o', markersize=5,
              color=color_0k, markeredgewidth=0.5)
    ax_B.axhline(0, linestyle="--", color="black")
    # Draw arrows from ancestral (x = 1) to 2K fitness (x = 2)
    for i, row in Repi.iterrows():
        if pd.notna(row["R"]) and pd.notna(row["M"]):
            start = (1, row["R"])
            end = (2, row["M"])
            arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=10,
                                    color=color_0k, lw=0.75)
            ax_B.add_patch(arrow)
    # Plot 2K fitness values (Tepi) at x = 2
    ax_B.plot(np.repeat(2, len(Tepi)), Tepi["M"], 'o', markersize=5,
              color=color_2k, markeredgewidth=0.5)
    # Draw arrows from 2K (x = 2) back to ancestral (x = 1)
    for i, row in Tepi.iterrows():
        if pd.notna(row["M"]) and pd.notna(row["R"]):
            start = (2, row["M"])
            end = (1, row["R"])
            arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=10,
                                    color=color_2k, lw=0.75)
            ax_B.add_patch(arrow)
    # Set ticks and custom labels for panel B
    ax_B.set_xticks([1.0, 2.0])
    ax_B.set_xticklabels(["0K", "2K"], fontsize=14)

    # Panel C: 2K to 15K
    ax_C.set_xlim(0.8, 2.2)
    ax_C.set_ylim(-0.15, 0.1)
    ax_C.set_ylabel(r'Fitness effect $(\Delta)$', fontsize=16)
    ax_C.tick_params(labelsize=14)
    # Plot 2K fitness values (Tepi) at x = 1
    ax_C.plot(np.repeat(1, len(Tepi)), Tepi["M"], 'o', markersize=5,
              color=color_2k, markeredgewidth=0.5)
    ax_C.axhline(0, linestyle="--", color="black")
    # Draw arrows from 2K (x = 1) to 15K (x = 2)
    for i, row in Tepi.iterrows():
        if pd.notna(row["M"]) and pd.notna(row["K"]):
            start = (1, row["M"])
            end = (2, row["K"])
            arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=10,
                                    color=color_2k, lw=0.75)
            ax_C.add_patch(arrow)
    # Plot 15K fitness values (Fepi) at x = 2
    ax_C.plot(np.repeat(2, len(Fepi)), Fepi["K"], 'o', markersize=5,
              color=color_15k, markeredgewidth=0.5)
    # Draw arrows from 15K (x = 2) back to 2K (x = 1)
    for i, row in Fepi.iterrows():
        if pd.notna(row["K"]) and pd.notna(row["M"]):
            start = (2, row["K"])
            end = (1, row["M"])
            arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=10,
                                    color=color_15k, lw=0.75)
            ax_C.add_patch(arrow)
    # Set ticks and custom labels for panel C
    ax_C.set_xticks([1.0, 2.0])
    ax_C.set_xticklabels(["2K", "15K"], fontsize=14)


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

    # Plot segben subfigures on axes B and C.
    create_segben(ax_top_middle, ax_top_right)

    # Pass the two axes to the plotting function.
    create_overlapping_dfes(ax_bottom_middle, ax_bottom_right)

    # Panel labels
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
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.tick_params(axis='both', width=1.5)
        ax.tick_params(axis='both', which='major', length=10, width=1.5, labelsize=14)
        ax.tick_params(axis='both', which='minor', length=5, width=1.6, labelsize=14)
    # Title A is special
    ax_top_middle.text(-1.1, 0.125, "A", fontsize=16, fontweight='heavy', va='top', ha='left')

    # Save the figure.
    output_dir = os.path.join('..', 'figs_paper')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "baym_results.svg")
    fig.savefig(output_path, format="svg", bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()
