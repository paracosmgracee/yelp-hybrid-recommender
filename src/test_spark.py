from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("TestJavaIntegration") \
    .master("local[*]") \
    .getOrCreate()

print("Java integration successful ✅")

spark.stop()