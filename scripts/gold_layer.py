import os

from pyspark.sql.functions import desc

from silver_layer import SILVER_PATH

GOLD_PATH = "file:///app/data/gold"


def process_gold_layer(spark, logger, df_silver=None):
    try:
        logger.info("GOLD LAYER PROCESSING")
        os.makedirs(os.path.dirname(GOLD_PATH.replace("file://", "")), exist_ok=True)

        if df_silver is None:
            df_silver = spark.read.format("delta").load(SILVER_PATH)

        feature_cols = [
            "Customer_ID",
            "Avg_Age",
            "Primary_Occupation",
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
            "Max_Credit_History_Age",
            "Primary_Credit_Mix",
            "Common_Payment_Behaviour",
            "Debt_to_Income_Ratio",
            "EMI_to_Income_Ratio",
            "Credit_Risk_Flag",
        ]

        df_gold = df_silver.select(*feature_cols).orderBy(
            desc("Credit_Risk_Flag"), desc("Debt_to_Income_Ratio")
        )

        os.makedirs("/app/data/gold", exist_ok=True)
        df_gold.write.format("delta").mode("overwrite").save(GOLD_PATH)
        df_gold.show(3)

        logger.info(f"Gold layer created with {df_gold.count()} records")
        return df_gold

    except Exception as e:
        logger.error(f"Gold layer processing failed: {str(e)}", exc_info=True)
        raise
