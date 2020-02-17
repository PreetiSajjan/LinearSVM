import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.regression.LabeledPoint;

import scala.Tuple2;

/**
 * Implementing sentiment analysis using classification by building 
 * Linear Support Vector Machine (SVM) model
 * @author Preeti Sajjan
 */
public class SVMModel {

	public static void main(String args[]) {

		System.setProperty("hadoop.home.dir", "C:/winutils");
		SparkConf sparkConf = new SparkConf()
				.setAppName("Linear Support Vector Machine (SVM) model")
				.setMaster("local[4]").set("spark.executor.memory", "1g"); //4 core processor to work individually with 1 gigabyte of heap memory
		JavaSparkContext sc = new JavaSparkContext(sparkConf);

		// relative path
		File textfile = new File("imdb_labelled.txt");
		
		// absolute path to our data file
		String path = textfile.getAbsolutePath();
		//reading the text file
		JavaRDD<String> file = sc.textFile(path);

		// Create a HashingTF instance to map email text to vectors of 10000 features
		final HashingTF tf = new HashingTF(10000);

		// file is split into words, and each word is mapped to one feature.
		// Create LabeledPoint datasets for positive and negative examples.
		JavaRDD<LabeledPoint> Examples = file.map(line -> {
			String[] tokens = line.split("\t");
			return new LabeledPoint(Integer.parseInt(tokens[1]), tf.transform(Arrays.asList(tokens[0].split(" "))));
		});

		// Split initial RDD into two... [60% training data, 40% testing data].
		JavaRDD<LabeledPoint> training = Examples.sample(false, 0.6, 11L);
		training.cache();        
		JavaRDD<LabeledPoint> test = Examples.subtract(training);       
		test.cache();

		// Create a Linear Support Vector Machine (SVM) model learner
		org.apache.spark.mllib.classification.SVMModel model = SVMWithSGD.train(training.rdd(), 1000);

		// Clear the default threshold.
		model.clearThreshold();

		// Compute raw scores on the test set.
		JavaPairRDD<Object, Object> scoreAndLabels = test.mapToPair(p ->
		new Tuple2<>(model.predict(p.features()), p.label()));
		
		//list holding 10 labels (sentiments) of test movie reviews 
		ArrayList<Object> labels = new ArrayList<Object>();
		scoreAndLabels.take(10).forEach(x -> {
			labels.add(x._2());
			});
		
		// Get evaluation metrics.
		BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(scoreAndLabels.rdd());

		// AUPRC - Area under precision-recall curve
		System.out.println("\nArea under precision-recall curve = " + metrics.areaUnderPR() + " (" + metrics.areaUnderPR()*100 + " %)\n");
		
		//Displaying the predictions
		System.out.println("\nFirst 10 scores: " + scoreAndLabels.take(10) + "\n");
		System.out.println("First 10 labels of test movie reviews: " + labels + "\n\n");
		
		// AUROC - Area under ROC 
		System.out.println("\nArea under ROC = " + metrics.areaUnderROC() + " (" + metrics.areaUnderROC()*100 + " %)\n");

		//closing the resource
		sc.close();

	}	
}