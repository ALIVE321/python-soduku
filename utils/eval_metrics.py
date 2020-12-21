from sklearn import metrics


def eval_metric(labels, predicts, classes):
    report = metrics.classification_report(labels, predicts, digits=4, labels=classes)
    # print("\n".join(report.split("\n")[-4:]))
    print(report)
