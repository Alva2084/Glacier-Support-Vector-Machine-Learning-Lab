from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


ID_LIKE_COLUMNS = {"POINTID", "X_Coor", "Y_Coor"}
TARGET_COLUMN = "label"

"""
Functions for loading data, selecting features, building pipelines, and training SVMs on the glacier dataset.
"""


@dataclass
class TrainingResult:
    modelName: str
    features: list[str]
    modelErrors: np.ndarray
    classificationReport: str
    balancedAccuracy: float
    receiverOperatingCharacteristicUnderCurve: float

# Load the glacier GIS dataset from CSV.
def loadData (csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)

# Return columns with a single value
def findConstantColumns (data: pd.DataFrame) -> list[str]:
    return [col for col in data.columns if data[col].nunique(dropna = False) <= 1]

# Return columns that start with "Unnamed"
def findUnnamedColumns (data: pd.DataFrame) -> list[str]:
    return [col for col in data.columns if col.startswith("Unnamed")]


# Return columns to use as features, excluding identifiers, target, and constant columns
def selectFeatureColumns (df: pd.DataFrame, targetColumn: str = TARGET_COLUMN) -> list[str]:

    dropColumns = set(findConstantColumns(df))
    dropColumns.update(findUnnamedColumns(df))
    dropColumns.update(ID_LIKE_COLUMNS)
    dropColumns.add(targetColumn)
    return [col for col in df.columns if col not in dropColumns]


# Take a stratified sample that preserves the class imbalance ratio
def stratifiedSample (
    data: pd.DataFrame,
    nRows: int,
    targetColumn: str = TARGET_COLUMN,
    random_state: int = 42,) -> pd.DataFrame:

    if nRows >= len(data):
        return data.copy()

    sampled = (
        data.groupby(targetColumn, group_keys = False)[list(data.columns)]
        .apply(
            lambda group: group.sample(
                max(1, int(round(nRows * len(group) / len(data)))),
                random_state = random_state,
            ),
            include_groups = False,
        )
        .reset_index(drop = True)
    )
    return sampled


# Build a pipeline with imputation, scaling, and either a linear or RBF SVM classifier
def buildPipeline (model: str = "linear") -> Pipeline:
    if model == "linear":
        estimator = LinearSVC(class_weight = "balanced", random_state = 42, max_iter = 5000)
    elif model == "rbf":
        estimator = SVC(kernel="rbf", class_weight="balanced", C = 10, gamma = "scale", random_state=42)
    else:
        raise ValueError("model must be 'linear' or 'rbf'")

    return Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy = "median")),
            ("scaler", StandardScaler()),
            ("svm", estimator),
        ]
    )

# Train an SVM and return evaluation metrics
def trainAndEvaluation (
    data: pd.DataFrame,
    model: str = "linear",
    sampleSaize: int | None = None,
    testSize: float = 0.2,
    randomState: int = 42,
) -> TrainingResult:
    working = data.copy()
    if sampleSaize is not None:
        working = stratifiedSample(working, nRows = sampleSaize, random_state = randomState)

    featureColumns = selectFeatureColumns(working)
    x = working[featureColumns]
    y = working[TARGET_COLUMN]

    xTrain, xTest, yTrain, yTest = train_test_split(
        x,
        y,
        test_size = testSize,
        stratify = y,
        random_state = randomState,
    )

    pipeline = buildPipeline(model = model)
    pipeline.fit(xTrain, yTrain)

    predictions = pipeline.predict(xTest)
    decisionScores = pipeline.decision_function(xTest)

    return TrainingResult(
        modelName = model,
        features = featureColumns,
        modelErrors = confusion_matrix(yTest, predictions),
        classificationReport = classification_report(yTest, predictions, digits = 4),
        balancedAccuracy = float(balanced_accuracy_score(yTest, predictions)),
        receiverOperatingCharacteristicUnderCurve = float(roc_auc_score(yTest, decisionScores)),
    )