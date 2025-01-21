from scipy import stats
import matplotlib.pyplot as plt
from Bio.SeqIO.FastaIO import SimpleFastaParser
import re
import pandas as pd
import seaborn as sns
import os
import numpy as np

#current working directory
sns.set_theme()
sns.set_context('paper')

repo = '../data/anurag_data'
metadata_path = repo + '/Metadata/'
datapath = repo + '/Analysis/Part_3_TnSeq_analysis/Processed_data_for_plotting'
figpath = '../figs_paper'
#names of libraries
libraries = ['REL606', 'REL607', 'REL11330', 'REL11333', 'REL11364', 'REL11336', 'REL11339', 'REL11389', 'REL11392',
             'REL11342', 'REL11345', 'REL11348', 'REL11367', 'REL11370']
#more interpretable names for the figures in the paper
libraries2 = ['Anc', 'Anc*', 'Ara–1', 'Ara–2', 'Ara–3', 'Ara–4', 'Ara–5', 'Ara–6', 'Ara+1', 'Ara+2', 'Ara+3', 'Ara+4',
              'Ara+5', 'Ara+6']

#opening the pandas file with all the metadata!
all_data = pd.read_csv(metadata_path + "all_metadata_REL606.txt", sep="\t")
names = all_data.iloc[:, 0]
gene_start = all_data.iloc[:, 3]
gene_end = all_data.iloc[:, 4]
strand = all_data.iloc[:, 5]
locations = np.transpose(np.vstack([gene_start, gene_end, strand]))
k12_tags = all_data.iloc[:, 2]
uniprot_rel606 = all_data.iloc[:, 6]

product = all_data.iloc[:, -1]

#list of genes to be excluded from analysis as they lie within large deletions
exclude_genes = np.loadtxt(repo + "/Analysis/Part_2_WGS_analysis/output/Deleted_genes_REL606_k12annotated.txt")
exclude_pseudogenes = np.loadtxt(
    repo + "/Analysis/Part_2_WGS_analysis/output/Deleted_pseudogenes_REL606_k12annotated.txt")
locations_pseudogenes = np.loadtxt(metadata_path + 'pseudogenes_locations_REL606.txt')
n_pseudo = exclude_pseudogenes.shape[1]

#fractions of the gene at the 5' and 3' ends to be excluded from analysis because they insertions there may not actually
#be disruptive to protein function
frac5p = 0.1
frac3p = 0.25

with open(metadata_path + "rel606_reference.fasta") as in_handle:
    for title, seq in SimpleFastaParser(in_handle):
        ta_sites = [m.start(0) for m in re.finditer('TA', seq)]
ta_sites = np.array(ta_sites)

#counting how many TA sites are present in each gene
ta_gene = np.zeros(len(names))
for i in range(0, len(names)):
    start = locations[i, 0]
    end = locations[i, 1]
    length = end - start
    #if the gene is on the forward strand
    if locations[i, 2] == 1:
        #counting sites only in the middle 80% of the gene, excluding 10% at each end
        ta_gene[i] = np.sum((ta_sites > start + length * frac5p) & (ta_sites < end - length * frac3p))
    elif locations[i, 2] == -1:
        ta_gene[i] = np.sum((ta_sites < start + length * frac5p) & (ta_sites > end - length * frac3p))

#loading the fitness data files now:
fitness_gene_corrected = np.load(datapath + '/fitness_corrected_genes.npy')
fitness_gene_relaxed = np.load(datapath + '/fitness_genes_relaxed_thresholds_updated.npy')
fitness_pseudogene = np.load(datapath + '/fitness_pseudogenes.npy')
#these are all slightly different metrics of error
error_gene_inv = np.load(datapath + '/errors_genes_inv.npy')
error_pseudogene_inv = np.load(datapath + '/errors_pseudogenes_inv.npy')
error_gene_hybrid = np.load(datapath + '/errors_genes_hybrid.npy')
error_pseudogene_hybrid = np.load(datapath + '/errors_pseudogenes_hybrid.npy')

#the dfe data pandas frame which is used for plotting a lot of the figure 2 panels
dfe_data = pd.read_csv(datapath + '/dfe_data_pandas.csv')

#renaming columns for purposes of the figure
dfe_data.rename(columns={'Fitness estimate': 'Fitness effect'}, inplace=True)

dfe_data_minus = dfe_data[(dfe_data['Ara Phenotype'] == 'Ara-')]
dfe_data_plus = dfe_data[(dfe_data['Ara Phenotype'] == 'Ara+')]
dfe_data = pd.concat([dfe_data_minus, dfe_data_plus])

pop_names_old = ['REL606', 'REL607', 'Ara-1', 'Ara-2', 'Ara-3', 'Ara-4', 'Ara-5', 'Ara-6', 'Ara+1', 'Ara+2', 'Ara+3',
                 'Ara+4', 'Ara+5', 'Ara+6']
for k in range(len(pop_names_old)):
    pop = pop_names_old[k]
    dfe_data.loc[(dfe_data['Population'] == pop) & (dfe_data['Mutator'] == 'non-mutator'), 'Population'] = libraries2[k]
    dfe_data.loc[(dfe_data['Population'] == pop) & (dfe_data['Mutator'] == 'mutator'), 'Population'] = libraries2[k]

dfe_data = dfe_data[~dfe_data['Population'].isin(['Ara–2', 'Ara+4'])]

sns.color_palette('CMRmap')

ridgeplot_palette = []
k_reordered = [0, 2, 4, 5, 6, 7, 1, 8, 9, 10, 12, 13]
for k in k_reordered:
    if k < 2:
        ridgeplot_palette.append(sns.color_palette('CMRmap')[2])
    elif k == 3 or k == 11:  ###note, these are Ara-2 and Ara+4 which we want to exclude from our analysis
        ridgeplot_palette.append(sns.color_palette('CMRmap')[1])
    else:
        ridgeplot_palette.append(sns.color_palette('CMRmap')[0])  #orange

dfe_ridgeplot_inverse_var = dfe_data.loc[(dfe_data['Fitness effect'] < 0.05) & (dfe_data['Fitness effect'] > -0.1)]

g = sns.FacetGrid(dfe_ridgeplot_inverse_var, row="Population", hue='Population', aspect=7, height=0.3,
                  palette=ridgeplot_palette)
sns.set_theme(context="paper", rc={"axes.facecolor": (0, 0, 0, 0)}, style='white')
g.map(sns.kdeplot, "Fitness effect",
      bw_adjust=.5, clip_on=True,
      fill=True, alpha=1, linewidth=1)
g.map(sns.kdeplot, "Fitness effect", clip_on=False, color="white", lw=0.5, bw_adjust=.25)
g.map(plt.axvline, x=0, color="white", lw=0.5, linestyle="dotted", clip_on=True)


# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0.03, .2, label, color='black', size=8,
            ha="left", va="center", transform=ax.transAxes)


g.map(label, "Fitness effect")

# Set the subplots to overlap
g.fig.subplots_adjust(hspace=-0.6)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[], ylabel='')
g.set(xticks=[-0.1, -0.05, 0, 0.05])
g.despine(left=True)
# g.set(xlim=(-0.2,0.1))
# g.set_xlabel('Fitness effect of insertion mutation', fontsize=10)

g.savefig(figpath + '/DFE_shape_compare.svg', format='svg', bbox_inches='tight')
