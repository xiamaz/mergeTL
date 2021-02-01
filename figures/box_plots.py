import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


def create_boxplot():
    df = pd.read_excel("F1_scores_k-fold.xlsx")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    plt.style.use("PaperDoubleFig.mplstyle")

    g1 = sns.boxplot(x="Dataset", y="Weighted_F1_score", hue="Protocol", data=df, showfliers=False, ax=axes[0])
    g1.set(title='weighted_F1 score')  # add a title
    g1.set(xlabel=None)  # remove the axis label
    g1.set(ylabel=None)

    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    g2 = sns.boxplot(x="Dataset", y="Avg_F1_score", hue="Protocol", data=df, showfliers=False, ax=axes[1])
    g2.set(title='average_F1 score')  # add a title
    g2.set(xlabel=None)  # remove the axis label
    g2.set(ylabel=None)

    plt.savefig('F1scores.pdf', dpi=300, bbox_inches='tight')