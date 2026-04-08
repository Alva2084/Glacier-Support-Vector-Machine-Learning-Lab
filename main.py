from pathlib import Path
from glacierSVM import loadData
from glacierSVM import trainAndEvaluation

"""
Do the thing
"""

def main():
    print("Starting SVM program...\n")

    csvPath = Path("glacier_land_ice102.csv")
    data = loadData(csvPath)

    linearResult = trainAndEvaluation(data, model = "linear", sampleSaize = 10_000)
    rbfResult = trainAndEvaluation(data, model = "rbf", sampleSaize = 5_000)

    print("=== Linear SVM ===")
    print("Features:", linearResult.features)
    print("Model Error:\n", linearResult.modelErrors)
    print(linearResult.classificationReport)
    print(f"Balanced accuracy: {linearResult.balancedAccuracy:.4f}")
    print(f"ROC AUC: {linearResult.receiverOperatingCharacteristicUnderCurve:.4f}")
    print("\n=== RBF SVM ===")
    print("Features:", rbfResult.features)
    print("Model Error:\n", rbfResult.modelErrors)
    print(rbfResult.classificationReport)
    print(f"Balanced accuracy: {rbfResult.balancedAccuracy:.4f}")
    print(f"Receiver Operating Characteristics (Under Curve): {rbfResult.receiverOperatingCharacteristicUnderCurve:.4f}")


if __name__ == "__main__":
    main()