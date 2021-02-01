import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import metrics
from collections import defaultdict
from argmagic import argmagic
from scipy import interp

from flowcat import io_functions, utils
from flowcat.classifier import SOMClassifier, som_dataset


def create_roc_results(trues, preds, output, model):
    """Create ROC and AUC metrics and save them to the given directory."""
    output.mkdir()
    curves = {}
    groups = model.config.groups
    auc = {}

    try:
        for i, group in enumerate(groups):
            curves[group] = metrics.roc_curve(trues[:, i], preds[:, i])

        for i, group in enumerate(groups):
            auc[group] = metrics.roc_auc_score(trues[:, i], preds[:, i])

        macro_auc = metrics.roc_auc_score(trues, preds, average="macro")
        micro_auc = metrics.roc_auc_score(trues, preds, average="micro")
        io_functions.save_json(
            {
                "one-vs-rest": auc,
                "macro": macro_auc,
                "micro": micro_auc,
            },
            output / "auc.json")
    except ValueError:
        curves[group] = metrics.roc_curve(trues[:, i], preds[:, i])
        io_functions.save_json(
            {
                "one-vs-rest": 0,
                "macro": 0,
                "micro": 0,
            }, output / "auc.json")
        pass

    return auc, curves


def compute_mean_ROC(curves, output):
    mean_fpr = np.linspace(0, 1, 100)
    tprs = defaultdict(list)

    for ele in curves:
        for group, scores in ele.items():
            tprs[group].append(interp(mean_fpr, scores[0], scores[1]))

    plt.style.use("PaperDoubleFig.mplstyle")
    fig, ax = plt.subplots()
    for group, tprs in tprs.items():
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        ax.plot(mean_fpr, mean_tpr,
                label=r'%s (AUC = %0.2f )' % (group, mean_auc))

    ax.plot((0, 1), (0, 1), "k--")
    ax.legend()
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC one-vs-rest")

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='best')

    fig.tight_layout()
    fig.savefig(str(output / "roc.png"), dpi=300)
    plt.close()


def main(data: utils.URLPath, kfold_dir: utils.URLPath, output: utils.URLPath):
    # dataset = io_functions.load_case_collection(data, meta)
    # dataset.set_data_path(utils.URLPath(""))

    dataset = som_dataset.SOMDataset.from_path(data)
    models = []
    dirs = next(os.walk(kfold_dir))[1]

    for dir in dirs:
        models.append(utils.URLPath(os.path.join(kfold_dir, dir)))

    aucs = []
    curves = []
    for i, model in enumerate(models):
        print(model)
        model = SOMClassifier.load(model)
        validate = model.get_validation_data(dataset)
        grps = validate.group_count
        groups = model.config.groups

        if len(grps.keys()) != len(groups):
            continue
        else:
            val_seq = model.create_sequence(validate)

            trues = np.concatenate([val_seq[i][1] for i in range(len(val_seq))])
            preds = np.array([p for p in model.model.predict_generator(val_seq)])

            auc, curve = create_roc_results(trues, preds, output / f"roc_n{i}", model)
            aucs.append(auc)
            curves.append(curve)

    compute_mean_ROC(curves, output)


if __name__ == "__main__":
    argmagic(main)
