import sagemaker

import os

from sagemaker.inputs import TrainingInput
from sagemaker.xgboost import XGBoost
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.functions import Join
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat,
)
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.model_metrics import ModelMetrics, MetricsSource


def get_pipeline(
    region: str,
    role: str,
    default_bucket: str = "ml-pipeline-project-aniolmg",
    pipeline_name: str = "TitanicPipeline",
):
    """Builds and returns a SageMaker Pipeline object."""

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    pipeline_session = PipelineSession(default_bucket=default_bucket)

    # ------------------------------------------------------
    # Parameters
    # ------------------------------------------------------
    bucket_param = ParameterString(name="Bucket", default_value=default_bucket)
    project_name_param = ParameterString(name="ProjectName", default_value="titanic")
    train_file_param = ParameterString(name="TrainFile", default_value="titanic_train.csv")
    test_file_param = ParameterString(name="TestFile", default_value="titanic_test.csv")

    # Hyperparameters
    max_depth_param = ParameterInteger(name="MaxDepth", default_value=8)
    eta_param = ParameterFloat(name="Eta", default_value=0.3)
    num_round_param = ParameterInteger(name="NumRound", default_value=200)
    objective_param = ParameterString(name="Objective", default_value="binary:logistic")
    target_param = ParameterString(name="Target")
    feature_columns_param = ParameterString(name="FeatureColumns")
    categorical_columns_param = ParameterString(name="CategoricalColumns")

    # ------------------------------------------------------
    # Paths
    # ------------------------------------------------------
    train_s3 = Join(on="/", values=["s3:/", bucket_param, "data", train_file_param])

    output_s3 = Join(on="/", values=["s3:/", bucket_param, project_name_param, "output"])

    # ------------------------------------------------------
    # Training Estimator
    # ------------------------------------------------------
    xgb_estimator = XGBoost(
        entry_point=os.path.join(CURRENT_DIR, "../training/train_model.py"),
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        framework_version="1.7-1",
        py_version="py3",
        output_path=output_s3,
        base_job_name="xgboost-train",
        sagemaker_session=pipeline_session,
        hyperparameters={
            "max_depth": max_depth_param,
            "eta": eta_param,
            "objective": objective_param,
            "num_round": num_round_param,
            "train_file": train_file_param,
            "target_column": target_param,
            "feature_columns": feature_columns_param,
            "categorical_columns": categorical_columns_param,
        },
    )

    # ------------------------------------------------------
    # Training Step
    # ------------------------------------------------------
    train_step = TrainingStep(
        name="TrainTitanicModel",
        estimator=xgb_estimator,
        inputs={"train": TrainingInput(train_s3, content_type="csv")},
    )

    # ------------------------------------------------------
    # Processing Step for Metrics
    # ------------------------------------------------------
    script_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve(
            framework="xgboost",
            region=region,
            version="1.7-1",
            py_version="py3",
        ),
        command=["python3"],
        instance_type="ml.t3.medium",
        instance_count=1,
        role=role,
        sagemaker_session=pipeline_session,
    )

    metrics_output_path = Join(
        on="/",
        values=[
            "s3:/",
            bucket_param,
            project_name_param,
            "metrics",
            ExecutionVariables.PIPELINE_EXECUTION_ID,
        ],
    )

    metrics_step = ProcessingStep(
        name="ComputeTitanicMetrics",
        processor=script_processor,
        code=os.path.join(CURRENT_DIR, "../processing/compute_metrics.py"),  # inside src/
        inputs=[
            ProcessingInput(
                source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=Join(on="/", values=["s3:/", bucket_param, "data", test_file_param]),
                destination="/opt/ml/processing/data",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="metrics",
                source="/opt/ml/processing/metrics",
                destination=metrics_output_path,
            )
        ],
        job_arguments=[
            "--input-model", "/opt/ml/processing/model",
            "--input-data", Join(on="/", values=["/opt/ml/processing/data", test_file_param]),
            "--output-metrics", "/opt/ml/processing/metrics",
            "--target_column", target_param,
            "--feature_columns", feature_columns_param,
            "--categorical_columns", categorical_columns_param,
        ],
    )

    # ------------------------------------------------------
    # Register Model
    # ------------------------------------------------------
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    metrics_step.properties.ProcessingOutputConfig.Outputs["metrics"].S3Output.S3Uri,
                    "metrics.json",
                ],
            ),
            content_type="application/json",
        )
    )

    register_step = RegisterModel(
        name="RegisterTitanicModel",
        estimator=xgb_estimator,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium"],
        transform_instances=["ml.m5.large"],
        model_package_group_name="TitanicModel",
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
        depends_on=[metrics_step],
    )

    # ------------------------------------------------------
    # Pipeline definition
    # ------------------------------------------------------
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            bucket_param, project_name_param,
            max_depth_param, eta_param, num_round_param, objective_param,
            train_file_param, test_file_param,
            target_param, feature_columns_param, categorical_columns_param,
        ],
        steps=[train_step, metrics_step, register_step],
        sagemaker_session=pipeline_session,
    )

    return pipeline