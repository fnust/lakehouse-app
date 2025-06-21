import os
from pyspark.sql.functions import col, avg, max, when
from pyspark.sql.types import DecimalType
from bronze_layer import BRONZE_PATH

SILVER_PATH = "file:///app/data/silver"


def process_silver_layer(spark, logger):
    try:
        logger.info("SILVER LAYER PROCESSING")
        os.makedirs(os.path.dirname(SILVER_PATH.replace("file://", "")), exist_ok=True)

        df_bronze = spark.read.format("delta").load(BRONZE_PATH)

        numeric_columns = [
            "Age",
            "Annual_Income",
            "Monthly_Inhand_Salary",
            "Num_Bank_Accounts",
            "Num_Credit_Card",
            "Interest_Rate",
            "Num_of_Loan",
            "Delay_from_due_date",
            "Num_of_Delayed_Payment",
            "Changed_Credit_Limit",
            "Num_Credit_Inquiries",
            "Outstanding_Debt",
            "Credit_Utilization_Ratio",
            "Credit_History_Age",
            "Total_EMI_per_month",
            "Amount_invested_monthly",
            "Monthly_Balance",
        ]

        for column in numeric_columns:
            df_bronze = df_bronze.withColumn(
                column,
                when(
                    col(column).cast("string").rlike("^\\d+\\.?\\d*$"),
                    col(column).cast("double"),
                ).otherwise(None),
            )

        df_filtered = df_bronze.filter(
            (col("Customer_ID").isNotNull())
            & (col("Annual_Income").isNotNull())
            & (col("Annual_Income") > 0)
            & (col("Age").isNotNull())
            & (col("Age") >= 18)
            & (col("Num_Bank_Accounts").isNotNull())
            & (col("Num_Credit_Card").isNotNull())
        )

        df_silver = df_filtered.groupBy("Customer_ID").agg(
            # Basic customer info
            max("Name").alias("Name"),
            avg("Age").cast("int").alias("Avg_Age"),
            max("Occupation").alias("Primary_Occupation"),
            # Financial aggregates
            avg("Annual_Income").cast(DecimalType(20, 2)).alias("Avg_Annual_Income"),
            avg("Monthly_Inhand_Salary")
            .cast(DecimalType(20, 2))
            .alias("Avg_Monthly_Salary"),
            # Credit behavior
            avg("Num_Bank_Accounts").cast("int").alias("Avg_Bank_Accounts"),
            avg("Num_Credit_Card").cast("int").alias("Avg_Credit_Cards"),
            avg("Interest_Rate").cast(DecimalType(20, 2)).alias("Avg_Interest_Rate"),
            avg("Num_of_Loan").cast("int").alias("Avg_Loans"),
            # Payment behavior
            avg("Delay_from_due_date").cast("int").alias("Avg_Delay_Days"),
            avg("Num_of_Delayed_Payment").cast("int").alias("Avg_Delayed_Payments"),
            # Credit utilization
            avg("Changed_Credit_Limit")
            .cast(DecimalType(20, 2))
            .alias("Avg_Credit_Limit_Change"),
            avg("Credit_Utilization_Ratio")
            .cast(DecimalType(20, 2))
            .alias("Avg_Credit_Utilization"),
            # Debt information
            avg("Outstanding_Debt")
            .cast(DecimalType(20, 2))
            .alias("Avg_Outstanding_Debt"),
            avg("Total_EMI_per_month")
            .cast(DecimalType(20, 2))
            .alias("Avg_Monthly_EMI"),
            # Credit history
            max("Credit_History_Age").alias("Max_Credit_History_Age"),
            # Most frequent values
            max("Credit_Mix").alias("Primary_Credit_Mix"),
            max("Payment_Behaviour").alias("Common_Payment_Behaviour"),
            max("Payment_of_Min_Amount").alias("Common_Min_Payment_Behavior"),
            # Target variable
            max("Credit_Score").alias("Primary_Credit_Score"),
        )

        df_silver = (
            df_silver.withColumn(
                "Debt_to_Income_Ratio",
                (col("Avg_Outstanding_Debt") / col("Avg_Annual_Income")).cast(
                    DecimalType(5, 2)
                ),
            )
            .withColumn(
                "EMI_to_Income_Ratio",
                (col("Avg_Monthly_EMI") / (col("Avg_Annual_Income") / 12)).cast(
                    DecimalType(5, 2)
                ),
            )
            .withColumn(
                "Credit_Risk_Flag",
                when(col("Primary_Credit_Score") == "Poor", 1)
                .when(col("Primary_Credit_Score") == "Standard", 2)
                .otherwise(3),
            )
        )

        silver_count = df_silver.count()
        logger.info(f"Silver layer contains {silver_count} customer records")

        df_silver.write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).save(SILVER_PATH)

        logger.info("Silver layer processing completed successfully")
        return df_silver

    except Exception as e:
        logger.error(f"Silver layer processing failed: {str(e)}", exc_info=True)
        raise
