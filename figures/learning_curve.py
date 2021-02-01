import pandas as pd
import matplotlib.pyplot as plt


# args: fname - excel file with accuray/F1 scores of all the runs
#       figname - name of the plot to save

def plot_line(fname, figname):
    data = pd.read_excel(fname)
    # change the name of the columns accordingly
    plt.style.use("PaperDoubleFig.mplstyle")
    lines = data.plot.line(x="#training samples per group", y=["f1_weighted_standalone", "f1_weighted_TL"],
                           figsize=(9, 7))
    lines.set_xlabel("number of samples per group")
    lines.set_ylabel("weighted f1-score")
    fig = lines.get_figure()
    fig.savefig(figname)


def main():
    json_file = "learning_curve_transition.xlsx"
    figname = "transition_weighted_with_withou_TL.png"

    plot_line(json_file, figname)


if __name__ == '__main__':
    main()
