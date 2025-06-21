import os

INPUT_DATA_PATH = "/app/data/input/train.csv"
BRONZE_PATH = "file:///app/data/bronze"


def process_bronze_layer(spark, logger):
    try:
        logger.info("BRONZE LAYER PROCESSING")
        logger.info(f"Loading raw data from {INPUT_DATA_PATH}")

        os.makedirs(os.path.dirname(BRONZE_PATH.replace("file://", "")), exist_ok=True)

        df_raw = (
            spark.read.option("header", "true")
            .option("inferSchema", "true")
            .csv(INPUT_DATA_PATH)
        )

        initial_count = df_raw.count()
        df_raw = df_raw.dropDuplicates()
        final_count = df_raw.count()

        if initial_count != final_count:
            logger.warning(f"Removed {initial_count - final_count} duplicate records")

        df_raw.show(3, truncate=False)

        # Write with compression and optimized partitioning
        df_raw.write.format("delta").option("mergeSchema", "true").option(
            "compression", "snappy"
        ).mode("overwrite").save(BRONZE_PATH)

        logger.info(f"Successfully saved {final_count} records to Bronze layer")
        return df_raw

    except Exception as e:
        logger.error(f"Bronze layer processing failed: {str(e)}", exc_info=True)
        raise
