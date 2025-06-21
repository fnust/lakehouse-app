import os
import logging
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
import mlflow
import mlflow.spark
from delta import configure_spark_with_delta_pip
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from bronze_layer import process_bronze_layer
from gold_layer import GOLD_PATH, process_gold_layer
from silver_layer import process_silver_layer


MLFLOW_URI = "file:///app/mlruns"

MODEL_NAME = "CustomerSpendingPredictor"
EXPERIMENT_NAME = "CreditScorePrediction"


def configure_logging():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename=f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    )
    logger = logging.getLogger("pipeline")
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    logging.getLogger("py4j").setLevel(logging.ERROR)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    return logger


def configure_spark():
    builder = (
        SparkSession.builder.appName(EXPERIMENT_NAME)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config(
            "spark.delta.logStore.class",
            "org.apache.spark.sql.delta.storage.HDFSLogStore",
        )
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def train_model(spark, logger, df_gold=None):
    try:
        logger.info("CREDIT RISK MODEL TRAINING")
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        if df_gold is None:
            df_gold = spark.read.format("delta").load(GOLD_PATH)

        feature_cols = [
            "Avg_Age",
            "Avg_Annual_Income",
            "Avg_Monthly_Salary",
            "Avg_Bank_Accounts",
            "Avg_Credit_Cards",
            "Avg_Interest_Rate",
            "Avg_Loans",
            "Avg_Delay_Days",
            "Avg_Delayed_Payments",
            "Avg_Credit_Limit_Change",
            "Avg_Credit_Utilization",
            "Avg_Outstanding_Debt",
            "Avg_Monthly_EMI",
            "Debt_to_Income_Ratio",
            "EMI_to_Income_Ratio",
        ]

        assembler = VectorAssembler(
            inputCols=feature_cols, handleInvalid="skip", outputCol="features"
        )

        # Convert Credit_Risk_Flag to string for classification
        label_indexer = StringIndexer(inputCol="Credit_Risk_Flag", outputCol="label")

        # Use RandomForest as it often works well for credit scoring
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=100,
            maxDepth=5,
            seed=42,
            impurity="gini",
        )

        pipeline = Pipeline(stages=[assembler, label_indexer, rf])

        train, test = df_gold.randomSplit([0.8, 0.2], seed=42)
        logger.info(f"Train samples: {train.count()}, Test samples: {test.count()}")

        with mlflow.start_run():
            logger.info("Training pipeline...")
            model = pipeline.fit(train)

            predictions = model.transform(test)

            evaluator = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction"
            )

            metrics = {
                "accuracy": evaluator.evaluate(
                    predictions, {evaluator.metricName: "accuracy"}
                ),
                "f1": evaluator.evaluate(predictions, {evaluator.metricName: "f1"}),
                "weightedPrecision": evaluator.evaluate(
                    predictions, {evaluator.metricName: "weightedPrecision"}
                ),
                "weightedRecall": evaluator.evaluate(
                    predictions, {evaluator.metricName: "weightedRecall"}
                ),
            }

            logger.info(
                f"Model metrics - Accuracy: {metrics['accuracy']:.2f}, F1: {metrics['f1']:.2f}, "
                f"Precision: {metrics['weightedPrecision']:.2f}, Recall: {metrics['weightedRecall']:.2f}"
            )

            rf_model = model.stages[-1]
            feature_importance = dict(
                zip(feature_cols, rf_model.featureImportances.toArray())
            )
            mlflow.log_dict(feature_importance, "feature_importance.json")

            mlflow.log_params(
                {
                    "features": feature_cols,
                    "target": "Credit_Risk_Flag",
                    "algorithm": "RandomForestClassifier",
                    "num_trees": 100,
                    "max_depth": 5,
                    "train_samples": train.count(),
                    "test_samples": test.count(),
                }
            )

            mlflow.log_metrics(metrics)

            mlflow.spark.log_model(
                model, "credit-risk-model", registered_model_name=MODEL_NAME
            )

            logger.info("Credit risk model training completed successfully")
            return model

    except Exception as e:
        logger.error(f"Model training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    logger = configure_logging()
    spark = None

    try:
        spark = configure_spark()
        logger.info("Spark session initialized")

        df_bronze = process_bronze_layer(spark, logger)
        df_silver = process_silver_layer(spark, logger)
        df_gold = process_gold_layer(spark, logger, df_silver)

        model = train_model(spark, logger, df_gold)

        logger.info("PIPELINE COMPLETED SUCCESSFULLY")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise

    finally:
        if spark:
            spark.stop()
            logger.info("Spark session stopped")
