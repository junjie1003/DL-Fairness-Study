import torch
from numpy import mean
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from fairlearn.metrics import MetricFrame, true_positive_rate, false_positive_rate, demographic_parity_difference, equalized_odds_difference


def get_metric_index():
    metric_index = ["acc", "bacc", "spd", "deo", "eod", "aaod", "aed", "time per epoch"]

    return metric_index


def _get_eo_frame(y_true, y_pred, sensitive_features, sample_weight) -> MetricFrame:
    fns = {"tpr": true_positive_rate, "fpr": false_positive_rate}
    sw_dict = {"sample_weight": sample_weight}
    sp = {"tpr": sw_dict, "fpr": sw_dict}
    eo = MetricFrame(
        metrics=fns,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        sample_params=sp,
    )
    return eo


def equal_opportunity_difference(y_true, y_pred, *, sensitive_features, method="between_groups", sample_weight=None) -> float:
    eo = _get_eo_frame(y_true, y_pred, sensitive_features, sample_weight)

    return eo.difference(method=method)["tpr"]


def average_odds_difference(y_true, y_pred, *, sensitive_features, method="between_groups", sample_weight=None) -> float:
    eo = _get_eo_frame(y_true, y_pred, sensitive_features, sample_weight)

    tprd = eo.difference(method=method)["tpr"]
    print(f"True Positive Rate Difference: {tprd}")
    fprd = eo.difference(method=method)["fpr"]
    print(f"False Positive Rate Difference: {fprd}")

    return 0.5 * (tprd + fprd)


def accuracy_equality_difference(y_true, y_pred, sensitive_features):
    misclassification_rate_p = sum(y_true[sensitive_features == 1] != y_pred[sensitive_features == 1]) / sum(sensitive_features == 1)
    misclassification_rate_n = sum(y_true[sensitive_features == 0] != y_pred[sensitive_features == 0]) / sum(sensitive_features == 0)
    return abs(misclassification_rate_p - misclassification_rate_n)


def multiclass_ovr_fairness(y_true, y_pred, sensitive_features):
    """[ovr]: Calculate metrics for the multiclass case using the one-vs-rest approach."""

    print("y_true:")
    print(y_true)
    print("y_pred:")
    print(y_pred)

    num_classes = len(torch.unique(y_true))
    print(f"num_classes: {num_classes}")

    if num_classes > 2:
        fairness_metrics = {}
        fairness_index = ["spd", "deo", "eod", "aaod", "aed"]
        for f_index in fairness_index:
            fairness_metrics[f_index] = []

        for cls in range(num_classes):
            # print(f"\nclass {cls}")

            # 1表示当前类别，0表示其他类别
            y_true_cls = (y_true == cls).to(torch.float)
            y_pred_cls = (y_pred == cls).to(torch.float)

            # print(f"y_true for class {cls}:")
            # print(y_true_cls)
            # print(f"y_pred for class {cls}:")
            # print(y_pred_cls)

            # Statistical Parity Difference (SPD)
            spd_cls = demographic_parity_difference(y_true_cls, y_pred_cls, sensitive_features=sensitive_features)
            fairness_metrics["spd"].append(spd_cls)

            # Equalized Odds Difference (DEO)
            deo_cls = equalized_odds_difference(y_true_cls, y_pred_cls, sensitive_features=sensitive_features)
            fairness_metrics["deo"].append(deo_cls)

            # Equal Opportunity Difference (EOD)
            eod_cls = equal_opportunity_difference(y_true_cls, y_pred_cls, sensitive_features=sensitive_features)
            fairness_metrics["eod"].append(eod_cls)

            # Average Absolute Odds Difference (AAOD)
            aaod_cls = average_odds_difference(y_true_cls, y_pred_cls, sensitive_features=sensitive_features)
            fairness_metrics["aaod"].append(aaod_cls)

            # Accuracy Equality Difference (AED)
            aed_cls = accuracy_equality_difference(y_true_cls, y_pred_cls, sensitive_features=sensitive_features)
            fairness_metrics["aed"].append(aed_cls)

        assert len(fairness_metrics["spd"]) == num_classes
        assert len(fairness_metrics["deo"]) == num_classes
        assert len(fairness_metrics["eod"]) == num_classes
        assert len(fairness_metrics["aaod"]) == num_classes
        assert len(fairness_metrics["aed"]) == num_classes

        spd = mean(fairness_metrics["spd"])
        deo = mean(fairness_metrics["deo"])
        eod = mean(fairness_metrics["eod"])
        aaod = mean(fairness_metrics["aaod"])
        aed = mean(fairness_metrics["aed"])

    else:
        spd = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
        deo = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)
        eod = equal_opportunity_difference(y_true, y_pred, sensitive_features=sensitive_features)
        aaod = average_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)
        aed = accuracy_equality_difference(y_true, y_pred, sensitive_features=sensitive_features)

    return spd, deo, eod, aaod, aed


def get_all_metrics(y_true, y_pred, sensitive_features):
    ret = {}

    # Accuracy
    ret["acc"] = accuracy_score(y_true, y_pred)
    # Balanced Accuracy
    ret["bacc"] = balanced_accuracy_score(y_true, y_pred)

    ret["spd"], ret["deo"], ret["eod"], ret["aaod"], ret["aed"] = multiclass_ovr_fairness(
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features
    )

    return ret


def print_all_metrics(ret):
    print("\nAccuracy: %.6f\n" % ret["acc"])
    print("Balanced Accuracy: %.6f\n" % ret["bacc"])

    print("Statistical Parity Difference (SPD): %.6f\n" % ret["spd"])
    print("Equalized Odds Difference (DEO): %.6f\n" % ret["deo"])
    print("Equal Opportunity Difference (EOD): %.6f\n" % ret["eod"])
    print("Average Absolute Odds Difference (AAOD): %.6f\n" % ret["aaod"])
    print("Accuracy Equality Difference (AED): %.6f\n" % ret["aed"])


if __name__ == "__main__":
    y_true = torch.tensor([0, 1, 1, 1, 2, 0, 2, 0, 1, 0, 1, 2])  # Example for 3-class classification
    y_pred = torch.tensor([2, 1, 1, 0, 2, 1, 1, 0, 0, 2, 0, 1])
    sf_data = torch.tensor([1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1])

    ret = get_all_metrics(y_true, y_pred, sensitive_features=sf_data)
    print_all_metrics(ret)
