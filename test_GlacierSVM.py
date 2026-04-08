# Adolfo Alvarez Jr

import pandas as pd
from sklearn.pipeline import Pipeline

from glacierSVM import (
    TARGET_COLUMN,
    buildPipeline,
    findConstantColumns,
    findUnnamedColumns,
    selectFeatureColumns,
    stratifiedSample,
    trainAndEvaluation,
)

"""
Unit tests for glacierSVM functions.
"""


def dataSample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2, 3, 4, 5],
            "POINTID": [10, 11, 12, 13, 14, 15],
            "X_Coor": [1, 2, 3, 4, 5, 6],
            "Y_Coor": [7, 8, 9, 10, 11, 12],
            "LST": [1.1, None, 1.3, 5.1, 5.2, 5.3],
            "band1": [10, 12, 11, 30, 28, 29],
            "band2": [9, 8, 10, 32, 31, 29],
            "ndsi": [-1, -1, -1, -1, -1, -1],
            TARGET_COLUMN: [0, 0, 0, 1, 1, 1],
        }
    )

# Test to see if findConstantColumns finds columns with only one unique value
def test_findConstantColumnsFindsPlaceholderColumns():
    data = dataSample()
    assert "ndsi" in findConstantColumns(data)


# Test to see if selectFeatureColumns correctly removes ID-like columns and the target column
def test_selectFeatureColumnsRemoves_ids_and_target():
    data = dataSample()
    features = selectFeatureColumns(data)
    assert features == ["LST", "band1", "band2"]


# Test to see if the pipeline is built correctly
def test_buildPipelineReturnsSklearnPipeline():
    pipeline = buildPipeline("linear")
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps.keys()) == ["imputer", "scaler", "svm"]


# Test to see if stratified sampling preserves class distribution
def test_stratifiedSamplePreservesClasses():
    data = pd.DataFrame({"feature": range(100), TARGET_COLUMN: [0] * 90 + [1] * 10})
    sample = stratifiedSample(data, nRows= 20, random_state = 42)
    counts = sample[TARGET_COLUMN].value_counts().to_dict()
    assert set(counts.keys()) == {0, 1}
    assert counts[1] > 0


# Test to see if the pipeline runs without errors and returns expected output structure
def test_trainAndEvaluateRuns():
    data = dataSample()
    result = trainAndEvaluation (data, model ="linear", sampleSaize = None, testSize = 0.33, randomState = 42)
    assert result.modelName == "linear"
    assert result.modelErrors.shape == (2, 2)
    assert 0.0 <= result.balancedAccuracy <= 1.0
    assert 0.0 <= result.receiverOperatingCharacteristicUnderCurve <= 1.0
