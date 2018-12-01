//Proyecto Final Kmeans-BisectingKMeans
//Chavez Perez Melissa #14212320
//Ingenieria Informatica

//Importación de librerias
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.clustering.BisectingKMeans
import org.apache.spark.sql.Column
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.KMeans
import org.apache.log4j._

//Inicio de sesion en spark
val spark = SparkSession.builder.master("local[*]").getOrCreate()
Logger.getLogger("org").setLevel(Level.ERROR)


//Cargar el archivo CSV
val df = spark.read.option("inferSchema","true").csv("Iris.csv").toDF("SepalLength", "SepalWidth", "PetalLength", "PetalWidth","class")
df.show()

//Cargar Etiqueta
val data = when($"class".contains("Iris-setosa"), 1.0).otherwise(when($"class".contains("Iris-virginica"), 3.0).otherwise(2.0))
val data2 = df.withColumn("etiqueta", data)

// Cargar el VectorAssembler
val assembler = new VectorAssembler().setInputCols(Array("SepalLength", "SepalWidth", "PetalLength", "PetalWidth","etiqueta")).setOutputCol("features")

//Transformacion de los datos

//---------------Algoritmo de K Means-----------//
val features = assembler.transform(data2)
features.show(5)
val kmeans = new KMeans().setK(3).setSeed(1L)
val model = kmeans.fit(features)


//--------Algoritmo de Bisecting K-KMeans------//
val features = assembler.transform(data2)
features.show(5)
val bkm = new BisectingKMeans().setK(2).setSeed(1)
val model = bkm.fit(features)


val WSSSE = model.computeCost(features)
println(s"Within set sum of Squared Errors = $WSSSE")
println("Cluster Centers: ")

val models = model.clusterCenters
models.foreach(println)
