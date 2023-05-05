from pyspark.sql import *
import pyspark
from delta import *
from pyspark.sql.functions import broadcast
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import col, lit, approx_count_distinct, concat, count, sum, when, countDistinct,rank
import plotly.graph_objs as go
import pyspark.pandas as ps



builder = pyspark.sql.SparkSession.builder.appName("MyApp") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.shuffle.compress", "true")\
    .config("spark.shuffle.spill.compress", "true")\
    .config("spark.executor.instances", 6) \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog").master("yarn")

spark = configure_spark_with_delta_pip(builder).getOrCreate()
# specify the paths of the folders containing the CSV files in HDFS
folder1_path = "hdfs:///dataset/output/part*"
folder2_path = "hdfs:///dataset/output2/small*"

# define the custom headers for each folder
folder1_header = "index,user_id,region_name,city_name,cpe_manufacturer_name,cpe_model_name,url_host,cpe_type_cd,cpe_model_os_type,price,date,part_of_day,request_cnt"
folder2_header = "user_id,age,is_male"

# define the schema for the CSV files
folder1_schema = StructType([StructField("index", IntegerType(), True),
                             StructField("user_id", IntegerType(), True),
                             StructField("region_name", StringType(), True),
                             StructField("city_name", StringType(), True),
                             StructField("cpe_manufacturer_name", StringType(), True),
                             StructField("cpe_model_name", StringType(), True),
                             StructField("url_host", StringType(), True),
                             StructField("cpe_type_cd", StringType(), True),
                             StructField("cpe_model_os_type", StringType(), True),
                             StructField("price", DoubleType(), True),
                             StructField("date", StringType(), True),
                             StructField("part_of_day", StringType(), True),
                             StructField("request_cnt", StringType(), True)])

folder2_schema = StructType([StructField("user_id", IntegerType(), True),
                             StructField("age", DoubleType(), True),
                             StructField("is_male", StringType(), True)])
#Partitioning not included if you check the rhs of the code
# read all CSV files in folder1 into a DataFrame
folder1_df = spark.read.format("csv").schema(folder1_schema).option("header", "false").load(folder1_path)#.repartition(100)
folder1_df = folder1_df.toDF(*folder1_header.split(","))

# read all CSV files in folder2 into a DataFrame
folder2_df = spark.read.format("csv").schema(folder2_schema).option("header", "false").load(folder2_path)# + "*.csv")
folder2_df = folder2_df.toDF(*folder2_header.split(","))
folder1_df.cache()
folder2_df.cache()

# perform a right join of the two DataFrames on the "user_id" column
joined_df = folder1_df.join(folder2_df, on=["user_id"], how="right")
#broadcast optimisation
#joined_df = folder1_df.join(broadcast(folder2_df), on=["user_id"], how="right")


# write the result to a new CSV file in HDFS
joined_df.write.format("csv").option("header", "true").mode("overwrite").save("hdfs://namenode:9000/dataset/output3")
from pyspark.sql.functions import sum 
from functools import reduce

def drop_null_rows(df):
    # Calculate the percentage of null values in each row
    num_columns = len(df.columns)
    # Expression to count null values in each column
    null_columns = (col(column).isNull().cast("int") for column in df.columns)

    # Add up null values across all columns
    null_count = reduce(lambda x, y: x + y, null_columns)
    null_percentage = (null_count / num_columns) * 100
    
    # Filter out rows with more than 40% null values
    filtered_df = df.filter(null_percentage <= 40)
    
    return filtered_df
filtered_df = drop_null_rows(joined_df)
for colm in joined_df.columns:
    col(colm).isNull().cast("int")
folder1_df.unpersist()
folder2_df.unpersist()
from pyspark.sql.functions import regexp_replace, col
#vectorization optimsation not included
'''
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

@udf(returnType=StringType())
def remove_tab(s):
    return s.replace("\t", "")

filtered_df = filtered_df.withColumn("request_cnt", remove_tab(col("request_cnt")))
filtered_df = filtered_df.withColumn("is_male", remove_tab(col("is_male")))
filtered_df = filtered_df.withColumn("request_cnt", col("request_cnt").cast("integer"))
filtered_df = filtered_df.withColumn("is_male", col("is_male").cast("integer"))
'''
# remove unwanted \t in the DataFrame
filtered_df = filtered_df.withColumn("request_cnt", regexp_replace(col("request_cnt"), "\t", ""))
filtered_df = filtered_df.withColumn("is_male", regexp_replace(col("is_male"), "\t", ""))
filtered_df = filtered_df.withColumn("request_cnt", col("request_cnt").cast("integer"))
filtered_df = filtered_df.withColumn("is_male", col("is_male").cast("integer"))
filtered_df.cache()
df=filtered_df
filtered_df.write.format("delta").option("header", "true").mode("overwrite").save("hdfs://namenode:9000/dataset/output3")
filtered_df=filtered_df.dropna()
filtered_df.groupBy('user_id').agg(approx_count_distinct('cpe_model_name').alias('distinct_models_count')).orderBy('distinct_models_count', ascending=False)
filtered_df.drop('distinct_models_count')
filtered_df.unpersist()
ml_df=filtered_df.drop_duplicates(['user_id','cpe_model_name']) 
ml_df=ml_df.drop('index','cpe_type_cd, cpe_model_os_type','date')
ml_df=ml_df.withColumn("age_bin", 
                          when(col("age") < 18, "0-17").\
                          when((col("age") >= 18) & (col("age") < 30), "18-29").\
                          when((col("age") >= 30) & (col("age") < 50), "30-49").\
                          when(col("age") >= 50, "50+"))
ml_df.cache()
def filter_by_optimized():
    
    manufacturer_counts = ml_df.groupBy('cpe_manufacturer_name','cpe_model_name', 'age_bin').agg(count('*').alias('count')).filter(col("age_bin").isNotNull()).orderBy(['age_bin', 'count'], ascending=[True, False])


    window = Window.partitionBy('age_bin').orderBy(col('count').desc())
    ranked_counts = manufacturer_counts.withColumn('rank', rank().over(window))

    top_models = ranked_counts.filter(col('rank') <= 3).drop('rank')

    total_count = top_models.agg(sum('count')).collect()[0][0]
    return(top_models,total_count)
def plotFunction(Iphone_11,Iphone_XR,Honor_10,Galaxy_A51,Total,Title):
    df = ps.DataFrame({'count': [Iphone_11, Iphone_XR, Honor_10, Galaxy_A51, Total-(Iphone_11+Iphone_XR+Honor_10+Galaxy_A51)]},
                  index=['Iphone_11', 'Iphone_XR', 'Honor_10', 'Galaxy_A51', 'Others'])

    data = [go.Pie(labels=df.index.tolist(), values=df['count'].tolist())]

    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height= 600,
        width= 600,
        title=go.layout.Title(text=Title)
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()
overall_market=filter_by_optimized()
plotFunction(2874,760,1294,260,overall_market[1],Title='Overall Market regardless of age')

indexer = StringIndexer(inputCols=["region_name","city_name","cpe_manufacturer_name","cpe_model_name"], outputCols=["region_name_indexed","city_name_indexed","cpe_manufacturer_name_indexed","cpe_model_name_indexed"])
encoder = OneHotEncoder(inputCols=["region_name_indexed","city_name_indexed","cpe_manufacturer_name_indexed","cpe_model_name_indexed"]\
                        , outputCols=["region_name_encoded","city_name_encoded","cpe_manufacturer_name_encoded","cpe_model_name_encoded"])
assembler = VectorAssembler(inputCols=["age","region_name_encoded","city_name_encoded","cpe_manufacturer_name_encoded","cpe_model_name_encoded","price","request_cnt","is_male"], outputCol="features")

pipeline = Pipeline(stages=[indexer, encoder, assembler])

transformed_df = pipeline.fit(ml_df).transform(ml_df).select("features")

correlation_matrix = Correlation.corr(transformed_df, "features")


##
indexer = StringIndexer(inputCols=["region_name","city_name","cpe_manufacturer_name","cpe_model_name"], outputCols=["region_name_indexed","city_name_indexed","cpe_manufacturer_name_indexed","cpe_model_name_indexed"])
encoder = OneHotEncoder(inputCols=["region_name_indexed","city_name_indexed","cpe_manufacturer_name_indexed","cpe_model_name_indexed"]\
                        , outputCols=["region_name_encoded","city_name_encoded","cpe_manufacturer_name_encoded","cpe_model_name_encoded"])
assembler = VectorAssembler(inputCols=["region_name_encoded","city_name_encoded","cpe_manufacturer_name_encoded","cpe_model_name_encoded","price","request_cnt","is_male"], outputCol="features")

pipeline = Pipeline(stages=[indexer, encoder, assembler])

data = pipeline.fit(ml_df).transform(ml_df)

train_data, test_data = data.randomSplit([0.75, 0.25])

lr = LinearRegression(featuresCol="features", labelCol="age",regParam=0.5)

model = lr.fit(train_data)
predictions = model.transform(test_data)
predicted_age=predictions.select('user_id','cpe_manufacturer_name','cpe_model_name','age','prediction')
predicted_age.cache()
ml_df.unpersist()
overall_market_need=overall_market[0]
overall_market_need.show()
overall_market_need.cache()
predicted_age.cache()
predicted_age = predicted_age.withColumn('age_bracket', (when((col('age').between(0, 17)),'0-17')
                    .when((col('age').between(18, 29)),'18-29')
                    .when((col('age').between(30, 49)), '30-49')
                    .when((col('age') >= 50), '50+')
                    .otherwise(None)))
suffix = '_r'
for col_name in overall_market_need.columns:
    new_col_name = col_name + suffix
    overall_market_need = overall_market_need.withColumnRenamed(col_name, new_col_name)
joined_df.cache()
# join the two dataframes using 'age_bracket' and 'age_bin' columns
joined_df = predicted_age.join(overall_market_need, (col('age_bracket') == col('age_bin_r')), 'left')
predicted_age.unpersist()
overall_market_need.unpersist()
#optimisation commented out
'''combined_col1 = concat(
    "user_id", "cpe_manufacturer_name", "cpe_model_name").alias("combined_col1")

joined_df = joined_df.select(combined_col1, *joined_df.columns)

combined_col2 = concat(
        "user_id",  "cpe_manufacturer_name_r", "cpe_model_name_r").alias("combined_col2")

joined_df = joined_df.select(combined_col2, *joined_df.columns)'''
from pyspark.sql.functions import concat_ws, col
joined_df = joined_df.withColumn("combined_col1", concat_ws("", col("user_id"), col("cpe_manufacturer_name"), col("cpe_model_name")))
joined_df = joined_df.withColumn("combined_col2", concat_ws("", col("user_id"), col("cpe_manufacturer_name_r"), col("cpe_model_name_r")))
joined_df = joined_df.filter(joined_df['combined_col1'] != joined_df['combined_col2'])
joined_df = joined_df.dropDuplicates(subset=["user_id"])
joined_df = joined_df.drop('combined_col1', 'combined_col2')
joined_df.write.format("delta").option("header", "true").mode("overwrite").save("hdfs://namenode:9000/dataset/output4")
joined_df.unpersist()
df = df.withColumn('age_bracket', (when((col('age').between(0, 17)),'0-17')
                    .when((col('age').between(18, 29)),'18-29')
                    .when((col('age').between(30, 49)), '30-49')
                    .when((col('age') >= 50), '50+')
                    .otherwise(None)))
# join the two dataframes using 'age_bracket' and 'age_bin' columns
df = df.join(overall_market_need, (col('age_bracket') == col('age_bin_r')), 'left')
df = df.withColumn("combined_col1", concat_ws("", col("index"), col("cpe_manufacturer_name"), col("cpe_model_name")))
df = df.withColumn("combined_col2", concat_ws("", col("index"), col("cpe_manufacturer_name_r"), col("cpe_model_name_r")))

#optimisations commented out
'''combined_col1 = concat(    "index", "cpe_manufacturer_name", "cpe_model_name").alias("combined_col1")

df = df.select(combined_col1, *df.columns)

combined_col2 = concat(
        "index",  "cpe_manufacturer_name_r", "cpe_model_name_r").alias("combined_col2")

df = df.select(combined_col2, *df.columns)'''

df = df.filter(df['combined_col1'] != df['combined_col2'])
df = df.dropDuplicates(subset=["index"])
df = df.drop('combined_col1', 'combined_col2','user_id','count_r')
df.write.format("delta").option("header", "true").mode("overwrite").save("hdfs://namenode:9000/dataset/output5")
