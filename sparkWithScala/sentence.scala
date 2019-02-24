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

val rdd_bert_reliable = df_reliable.select("index","label","title","sentence").as[SequenceRecord]
.rdd.map(r => (r.title+:r.sentence.slice(0,k):+"1").mkString("\t")+"\n")

val rdd_bert_fake = df_fake.select("index","label","title","sentence").as[SequenceRecord]
.rdd.map(r => (r.title+:r.sentence.slice(0,k):+"0").mkString("\t")+"\n")

rdd_bert_reliable.repartition(1).saveAsTextFile(textPath+"reliable")
rdd_bert_fake.repartition(1).saveAsTextFile(textPath+"fake")



