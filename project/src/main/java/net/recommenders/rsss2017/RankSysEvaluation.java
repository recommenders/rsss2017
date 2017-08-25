package net.recommenders.rsss2017;

import es.uam.eps.ir.ranksys.core.feature.FeatureData;
import es.uam.eps.ir.ranksys.core.feature.SimpleFeatureData;
import es.uam.eps.ir.ranksys.core.preference.ConcatPreferenceData;
import es.uam.eps.ir.ranksys.core.preference.PreferenceData;
import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;
import es.uam.eps.ir.ranksys.fast.index.SimpleFastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.SimpleFastUserIndex;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import es.uam.eps.ir.ranksys.metrics.RecommendationMetric;
import es.uam.eps.ir.ranksys.metrics.SystemMetric;
import es.uam.eps.ir.ranksys.metrics.basic.AverageRecommendationMetric;
import es.uam.eps.ir.ranksys.metrics.basic.NDCG;
import es.uam.eps.ir.ranksys.metrics.basic.Precision;
import es.uam.eps.ir.ranksys.metrics.basic.Recall;
import es.uam.eps.ir.ranksys.metrics.rank.NoDiscountModel;
import es.uam.eps.ir.ranksys.metrics.rank.RankingDiscountModel;
import es.uam.eps.ir.ranksys.metrics.rel.BinaryRelevanceModel;
import es.uam.eps.ir.ranksys.metrics.rel.NoRelevanceModel;
import es.uam.eps.ir.ranksys.nn.user.UserNeighborhoodRecommender;
import es.uam.eps.ir.ranksys.nn.user.neighborhood.TopKUserNeighborhood;
import es.uam.eps.ir.ranksys.nn.user.neighborhood.UserNeighborhood;
import es.uam.eps.ir.ranksys.nn.user.sim.UserSimilarity;
import es.uam.eps.ir.ranksys.nn.user.sim.VectorCosineUserSimilarity;
import es.uam.eps.ir.ranksys.rec.Recommender;
import es.uam.eps.ir.ranksys.rec.runner.RecommenderRunner;
import es.uam.eps.ir.ranksys.rec.runner.fast.FastFilterRecommenderRunner;
import es.uam.eps.ir.ranksys.rec.runner.fast.FastFilters;
import net.recommenders.rival.examples.DataDownloader;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;
import org.jooq.lambda.Unchecked;
import org.ranksys.formats.feature.SimpleFeaturesReader;
import org.ranksys.formats.index.ItemsReader;
import org.ranksys.formats.index.UsersReader;
import org.ranksys.formats.preference.SimpleRatingPreferencesReader;
import org.ranksys.formats.rec.RecommendationFormat;
import org.ranksys.formats.rec.SimpleRecommendationFormat;

import java.io.*;
import java.util.*;
import java.util.function.Function;
import java.util.function.IntPredicate;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import static org.ranksys.formats.parsing.Parsers.lp;


public class RankSysEvaluation {
    public static void main(String[] args) throws IOException {
        prepareData();
        String path = "movielens100k/ml-100k/";
        Map<String, Supplier<Recommender<Long, Long>>> recMap = new HashMap<>();

        String userPath = path + "testusers.u";
        String itemPath = path + "testitems.i";
        String trainDataPath = path + "u1.base";
        String testDataPath = path + "u1.test";
        String[] paths = new String[2];
        paths[0] = trainDataPath;
        paths[1] = testDataPath;
        prepareUserItemFiles(userPath, itemPath, paths);

        FastUserIndex<Long> userIndex = SimpleFastUserIndex.load(UsersReader.read(userPath, lp));
        FastItemIndex<Long> itemIndex = SimpleFastItemIndex.load(ItemsReader.read(itemPath, lp));
        FastPreferenceData<Long, Long> trainData = SimpleFastPreferenceData.load(SimpleRatingPreferencesReader.get().read(trainDataPath, lp, lp), userIndex, itemIndex);
        FastPreferenceData<Long, Long> testData = SimpleFastPreferenceData.load(SimpleRatingPreferencesReader.get().read(testDataPath, lp, lp), userIndex, itemIndex);

        recMap.put("ub", () -> {
            /**
             * Change the values below
             */
            double alpha = 1.0;
            int k = 50;
            int q = 1;

            UserSimilarity<Long> sim = new VectorCosineUserSimilarity<>(trainData, alpha, true);
            UserNeighborhood<Long> neighborhood = new TopKUserNeighborhood<>(sim, k);
            return new UserNeighborhoodRecommender<>(trainData, neighborhood, q);
        });

        /**
         * Change the values below
         */
        int cutoff = 10;
        double threshold = 5.0;
        int maxlength = 100;
        evaluate(recMap, userIndex, itemIndex, trainData, testData, cutoff, threshold, maxlength);

        /**
         * Change the values below
         */
        cutoff = 5;
        threshold = 3.0;
        maxlength = 5;
        evaluate(recMap, userIndex, itemIndex, trainData, testData, cutoff, threshold, maxlength);
    }

    private static void prepareData() {
        String movielens10kURL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip";
        String folder = "movielens100k";
        DataDownloader dd = new DataDownloader(movielens10kURL, folder);
        dd.downloadAndUnzip();
    }

    private static void evaluate(Map<String, Supplier<Recommender<Long, Long>>> recMap, FastUserIndex<Long> userIndex, FastItemIndex<Long> itemIndex, FastPreferenceData<Long, Long> trainData, FastPreferenceData<Long, Long> testData, int cutoff, double threshold, int maxLength) {
        Set<Long> targetUsers = testData.getUsersWithPreferences().collect(Collectors.toSet());
        RecommendationFormat<Long, Long> format = new SimpleRecommendationFormat<>(lp, lp);
        Function<Long, IntPredicate> filter = FastFilters.all();
        RecommenderRunner<Long, Long> runner = new FastFilterRecommenderRunner<>(userIndex, itemIndex, targetUsers.stream(), filter, maxLength);

        recMap.forEach(Unchecked.biConsumer((name, recommender) -> {
            System.out.println("\nRunning " + name);
            try (RecommendationFormat.Writer<Long, Long> writer = format.getWriter(name)) {
                runner.run(recommender.get(), writer);
            }
        }));

        PreferenceData<Long, Long> totalData = new ConcatPreferenceData<>(trainData, testData);

        // EVALUATED AT CUTOFF 10
        // BINARY RELEVANCE
        BinaryRelevanceModel<Long, Long> binRel = new BinaryRelevanceModel<>(false, testData, threshold);
        // NO RELEVANCE
        NoRelevanceModel<Long, Long> norel = new NoRelevanceModel<>();
        // NO RANKING DISCOUNT
        RankingDiscountModel disc = new NoDiscountModel();


        Map<String, SystemMetric<Long, Long>> sysMetrics = new HashMap<>();

        ////////////////////////
        // INDIVIDUAL METRICS //
        ////////////////////////
        Map<String, RecommendationMetric<Long, Long>> recMetrics = new HashMap<>();

        // PRECISION
        recMetrics.put("prec@" + cutoff, new Precision<>(cutoff, binRel));
        // RECALL
        recMetrics.put("recall@" + cutoff, new Recall<>(cutoff, binRel));
        // nDCG
        recMetrics.put("ndcg@"+ cutoff, new NDCG<>(cutoff, new NDCG.NDCGRelevanceModel<>(false, testData, threshold)));

        // AVERAGE VALUES OF RECOMMENDATION METRICS FOR ITEMS IN TEST
        int numUsers = testData.numUsersWithPreferences();
        recMetrics.forEach((name, metric) -> sysMetrics.put(name, new AverageRecommendationMetric<>(metric, numUsers)));

        int numItems = totalData.numItemsWithPreferences();

        recMap.forEach(Unchecked.biConsumer((name, recommender) -> {

            format.getReader(name).readAll().forEach(rec -> sysMetrics.values().forEach(metric -> metric.add(rec)));
        }));
        sysMetrics.forEach((name, metric) -> System.out.println(name + "\t" + metric.evaluate()));
    }

    private static void prepareUserItemFiles(String userPath, String itemPath, String[] dataPaths) throws IOException {
        Set<String> users = new HashSet<>();
        Set<String> items = new HashSet<>();
        for (String path: dataPaths) {
            LineIterator it = FileUtils.lineIterator(new File(path), "UTF-8");
            try {
                while (it.hasNext()) {
                    String[] line = it.nextLine().split("\t");
                    String user = line[0];
                    String item = line[1];
                    users.add(user);
                    items.add(item);
                }
            } finally {
                it.close();
            }
        }
        writeSetToFile(userPath, users);
        writeSetToFile(itemPath,items);
    }

    private static void writeSetToFile(String filePath, Set<String> elements) throws FileNotFoundException {
        PrintStream out = new PrintStream(new FileOutputStream(filePath));
        Iterator hashSetIterator = elements.iterator();
        while(hashSetIterator.hasNext()){
            out.println(hashSetIterator.next());
        }
    }

}
