import org.apache.spark.ml._
import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT}
import com.johnsnowlabs.nlp.{Annotation, _}
import com.johnsnowlabs.nlp.annotator.{NerConverter, SentenceDetector, Tokenizer}
import com.johnsnowlabs.util.PipelineModels
import org.apache.spark.sql.{Dataset, Row}

val documentAssembler = new DocumentAssembler().setInputCol("content").setOutputCol("document")
val sentenceDetector = new SentenceDetector().setInputCols("document").setOutputCol("sentence")

val pipeline = new Pipeline().setStages(
    Array(documentAssembler,
      sentenceDetector
    ))

val ACCESS_KEY = "AKIAIOQ5GEHHQ47FJZFQ"
// Encode the Secret Key as that can contain "/"
val SECRET_KEY = "27URHQOXwXkUoeV2VOE0DoLxEAeTnpccfcpeZ86a"
val ENCODED_SECRET_KEY = SECRET_KEY.replace("/", "%2F")
val BUCKET_NAME = "wenxinfakenews"

//read data from s3
val df = spark.read.option("header",true).csv(s"s3a://$ACCESS_KEY:$SECRET_KEY@$BUCKET_NAME/news_concat.csv")
//val df = spark.read.option("header",true).csv("/FileStore/tables/*")
val colNames = Seq("index","label","content","title")
val dfRenamed = df.toDF(colNames:_*).na.drop

val model = pipeline.fit(dfRenamed)
val dfTransformed = model.transform(dfRenamed)
case class Sentence(annotatorType: String, begin: Int, end: Int, result: String, metadata: Map[String, String])
case class Record(index: String, label: String, title: String, sentence: Array[Sentence])
val dfSentence = dfTransformed.select("index","label","title","sentence").as[Record]
.rdd.map(r => 
         (r.index, 
          r.label, 
          r.title,
          r.sentence.map(s => s.result),
          r.sentence.slice(0,3).map(s=>s.result)))
.toDF(Seq("index","label","title","sentence","first3"):_*)

//write to s3 in format parquet, can read result again and change format later

dfSentence.write.parquet(s"s3a://$ACCESS_KEY:$SECRET_KEY@$BUCKET_NAME/news_array")

//convert to bert input format
val parquetPath = s"s3a://$ACCESS_KEY:$SECRET_KEY@$BUCKET_NAME/news_array/"
val textPath = s"s3a://$ACCESS_KEY:$SECRET_KEY@$BUCKET_NAME/news_text/"
val df_fromParquet = spark.read.parquet(parquetPath)

case class SequenceRecord(index: String, label: String, title: String, sentence: Seq[String])

val k = 3

val df_reliable = df_fromParquet.filter($"label"==="reliable")
val df_fake = df_fromParquet.filter($"label"=!="reliable")

case class SequenceRecord(index: String, label: String, title: String, sentence: Seq[String])

val k = 3

val df_reliable = df_fromParquet.filter($"label"==="reliable")
val df_fake = df_fromParquet.filter($"label"=!="reliable")

val df_reliable_transformed = df_reliable.select("index","label","title","sentence").as[SequenceRecord]
.rdd.map(r => 
         (((r.title+".")+:r.sentence.slice(0,k)).mkString("").replaceAll("\t+", " ").replaceAll("\n+", "").replaceAll("\\.+", "."),1))
.toDF("text","label")

val df_fake_transformed = df_fake.select("index","label","title","sentence").as[SequenceRecord]
.rdd.map(r =>  
         (((r.title+".")+:r.sentence.slice(0,k)).mkString("").replaceAll("\t+", " ").replaceAll("\n+", "").replaceAll("\\.+", "."),0))
.toDF("text","label")

df_reliable_transformed.repartition(1).write.option("delimiter","\t").mode("overwrite").csv(s"s3a://$ACCESS_KEY:$ENCODED_SECRET_KEY@$BUCKET_NAME/news_text_tsv/reliable")
//.saveAsTextFile(s"s3a://$ACCESS_KEY:$ENCODED_SECRET_KEY@$BUCKET_NAME/news_text_cui1/reliable")
df_fake_transformed.repartition(1).write.option("delimiter","\t").mode("overwrite").csv(s"s3a://$ACCESS_KEY:$ENCODED_SECRET_KEY@$BUCKET_NAME/news_text_tsv/fake")
//.saveAsTextFile(s"s3a://$ACCESS_KEY:$ENCODED_SECRET_KEY@$BUCKET_NAME/news_text_cui1/fake")
case class SequenceRecord(index: String, label: String, title: String, sentence: Seq[String])

val k = 3

val df_reliable = df_fromParquet.filter($"label"==="reliable")
val df_fake = df_fromParquet.filter($"label"=!="reliable")

val df_reliable_transformed = df_reliable.select("index","label","title","sentence").as[SequenceRecord]
.rdd.map(r => 
         (((r.title+".")+:r.sentence.slice(0,k)).mkString("").replaceAll("\t+", " ").replaceAll("\n+", "").replaceAll("\\.+", "."),1))
.toDF("text","label")

val df_fake_transformed = df_fake.select("index","label","title","sentence").as[SequenceRecord]
.rdd.map(r =>  
         (((r.title+".")+:r.sentence.slice(0,k)).mkString("").replaceAll("\t+", " ").replaceAll("\n+", "").replaceAll("\\.+", "."),0))
.toDF("text","label")

df_reliable_transformed.repartition(1).write.option("delimiter","\t").mode("overwrite").csv(s"s3a://$ACCESS_KEY:$ENCODED_SECRET_KEY@$BUCKET_NAME/news_text_tsv/reliable")
//.saveAsTextFile(s"s3a://$ACCESS_KEY:$ENCODED_SECRET_KEY@$BUCKET_NAME/news_text_cui1/reliable")
df_fake_transformed.repartition(1).write.option("delimiter","\t").mode("overwrite").csv(s"s3a://$ACCESS_KEY:$ENCODED_SECRET_KEY@$BUCKET_NAME/news_text_tsv/fake")
//.saveAsTextFile(s"s3a://$ACCESS_KEY:$ENCODED_SECRET_KEY@$BUCKET_NAME/news_text_cui1/fake")

val test = splits_reliable(0).union(splits_fake(0))
val validate = splits_reliable(1).union(splits_fake(1))
val train = splits_reliable(2).union(splits_fake(2))

test.repartition(1).write.option("delimiter","\t").mode("overwrite").csv(s"s3a://$ACCESS_KEY:$ENCODED_SECRET_KEY@$BUCKET_NAME/splits/news_tsv/test")
validate.repartition(1).write.option("delimiter","\t").mode("overwrite").csv(s"s3a://$ACCESS_KEY:$ENCODED_SECRET_KEY@$BUCKET_NAME/splits/news_tsv/validate")
train.repartition(1).write.option("delimiter","\t").mode("overwrite").csv(s"s3a://$ACCESS_KEY:$ENCODED_SECRET_KEY@$BUCKET_NAME/splits/news_tsv/train")



