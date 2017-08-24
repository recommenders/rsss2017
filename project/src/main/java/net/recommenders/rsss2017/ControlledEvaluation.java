package net.recommenders.rsss2017;

import java.io.File;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import net.recommenders.rival.core.DataModelFactory;
import net.recommenders.rival.core.DataModelIF;
import net.recommenders.rival.core.DataModelUtils;
import net.recommenders.rival.core.Parser;
import net.recommenders.rival.core.SimpleParser;
import net.recommenders.rival.core.TemporalDataModelIF;
import net.recommenders.rival.evaluation.metric.ranking.AbstractRankingMetric;
import net.recommenders.rival.evaluation.metric.ranking.NDCG;
import net.recommenders.rival.evaluation.metric.ranking.Precision;
import net.recommenders.rival.evaluation.metric.ranking.Recall;
import net.recommenders.rival.evaluation.statistics.EffectSize;
import net.recommenders.rival.evaluation.statistics.StatisticalSignificance;
import net.recommenders.rival.evaluation.strategy.EvaluationStrategy;
import net.recommenders.rival.evaluation.strategy.RelPlusN;
import net.recommenders.rival.evaluation.strategy.UserTest;
import net.recommenders.rival.examples.DataDownloader;
import net.recommenders.rival.recommend.frameworks.AbstractRunner;
import net.recommenders.rival.recommend.frameworks.RecommendationRunner;
import net.recommenders.rival.recommend.frameworks.lenskit.LenskitRecommenderRunner;
import net.recommenders.rival.recommend.frameworks.mahout.MahoutRecommenderRunner;
import net.recommenders.rival.recommend.frameworks.ranksys.RanksysRecommenderRunner;
import net.recommenders.rival.split.parser.MovielensParser;
import net.recommenders.rival.split.splitter.RandomSplitter;
import net.recommenders.rival.split.splitter.Splitter;

/**
 *
 * @author Alejandro
 */
public class ControlledEvaluation {

    public static void main(String[] args) throws Exception {
        // Take a dataset
        String url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip";
        String folder = "data/ml-100k";
        String fullDataset = folder + "/ml-100k/u.data";
        if (!new File(fullDataset).exists()) {
            System.out.println("Downloading " + url);
            DataDownloader dd = new DataDownloader(url, folder);
            dd.downloadAndUnzip();
        }
        Parser<Long, Long> parser = new MovielensParser();
        TemporalDataModelIF<Long, Long> data = parser.parseTemporalData(new File(fullDataset));
        // Split: 80-20 random per user
        Splitter<Long, Long> splitter = new RandomSplitter<>(0.8f, true, 1L, false);
        TemporalDataModelIF<Long, Long>[] split = splitter.split(data);
        String trainSplitFile = "split_train__rnd80_pu_npi.dat";
        DataModelUtils.saveDataModel(split[0], trainSplitFile, false, "\t");
        String testSplitFile = "split_test__rnd80_pu_npi.dat";
        DataModelUtils.saveDataModel(split[1], testSplitFile, false, "\t");
        // Generate recommendations (svd50, ubknn50):
        Set<String> recs = new HashSet<>();
        for (String method : new String[]{"ubknn50", "svd50"}) {
            AbstractRunner<Long, Long> rec = null;
            //   - lenskit
            Properties lenskitProperties = getLenskitProperties(method);
            lenskitProperties.put(RecommendationRunner.TRAINING_SET, trainSplitFile);
            lenskitProperties.put(RecommendationRunner.OUTPUT, ".");
            rec = new LenskitRecommenderRunner(lenskitProperties);
            rec.run(AbstractRunner.RUN_OPTIONS.OUTPUT_RECS, split[0], split[1]);
            String lenskitRec = rec.getCanonicalFileName();
            recs.add(lenskitRec);
            //   - ranksys
            Properties ranksysProperties = getRanksysProperties(method);
            ranksysProperties.put(RecommendationRunner.TRAINING_SET, trainSplitFile);
            ranksysProperties.put(RecommendationRunner.OUTPUT, ".");
            rec = new RanksysRecommenderRunner(ranksysProperties);
            rec.run(AbstractRunner.RUN_OPTIONS.OUTPUT_RECS, split[0], split[1]);
            String ranksysRec = rec.getCanonicalFileName();
            recs.add(ranksysRec);
            //   - mahout
            Properties mahoutProperties = getMahoutProperties(method);
            mahoutProperties.put(RecommendationRunner.TRAINING_SET, trainSplitFile);
            mahoutProperties.put(RecommendationRunner.OUTPUT, ".");
            rec = new MahoutRecommenderRunner(mahoutProperties);
            rec.run(AbstractRunner.RUN_OPTIONS.OUTPUT_RECS, split[0], split[1]);
            String mahoutRec = rec.getCanonicalFileName();
            recs.add(mahoutRec);
        }
        // Select candidate items for evaluation: RelPlusN, N=100 (actually, an approximation) vs UserTest
        Set<String> recsToEvaluate = new HashSet<>();
        for (String recName : recs) {
            for (EvaluationStrategy<Long, Long> strategy : new EvaluationStrategy[]{
                new RelPlusN(split[0], split[1], 100, 5.0, 1L),
                new UserTest(split[0], split[1], 5.0)
            }) {
                String r = recName + "__" + strategy + ".dat";
                recsToEvaluate.add(r);
                if (new File(r).exists()) {
                    System.out.println("File " + r + " already exists. Skipping");
                    continue;
                }
                // load the recommender in memory
                DataModelIF<Long, Long> recModel = new SimpleParser().parseData(new File(recName.replace(".stats", "")));
                DataModelIF<Long, Long> modelToEval = DataModelFactory.getDefaultModel();
                for (Long user : recModel.getUsers()) {
                    for (Long item : strategy.getCandidateItemsToRank(user)) {
                        double s = recModel.getUserItemPreference(user, item);
                        if (!Double.isNaN(s)) {
                            modelToEval.addPreference(user, item, s);
                        }
                    }
                }
                DataModelUtils.saveDataModel(modelToEval, r, false, "\t");
            }
        }
        // Compute evaluation metrics: ndcg@5, ndcg@10, P@5, P@10, R@5, R@10
        int[] cutoffs = new int[]{5, 10};
        for (String r : recsToEvaluate) {
            DataModelIF<Long, Long> test = split[1];
            DataModelIF<Long, Long> evalModel = new SimpleParser().parseData(new File(r));
            System.out.println(r);
            for (AbstractRankingMetric<Long, Long> m : new AbstractRankingMetric[]{
                new NDCG<>(evalModel, test, 5.0, cutoffs, NDCG.TYPE.EXP),
                new Precision<>(evalModel, test, 5.0, cutoffs),
                new Recall<>(evalModel, test, 5.0, cutoffs),}) {
                m.compute();
                for (int cutoff : cutoffs) {
                    System.out.println("\t" + m + "@" + cutoff + "=" + m.getValueAt(cutoff));
                }
            }
        }
        // Statistical testing: paired t-test, Wilcoxon, EffectSize
        //     note: here we are comparing all vs all algorithms, 
        //     usually one (or some) of them is "special" (the baseline(s) against we want to compare)
        for (String recName1 : recs) {
            for (String recName2 : recs) {
                if (recName1.compareTo(recName2) >= 0) {
                    // let's assume comparisons are symmetric
                    continue;
                }
                for (EvaluationStrategy<Long, Long> strategy : new EvaluationStrategy[]{
                    new RelPlusN(split[0], split[1], 100, 5.0, 1L),
                    new UserTest(split[0], split[1], 5.0)
                }) {
                    DataModelIF<Long, Long> test = split[1];
                    String r1 = recName1 + "__" + strategy + ".dat";
                    DataModelIF<Long, Long> evalModel1 = new SimpleParser().parseData(new File(r1));
                    String r2 = recName2 + "__" + strategy + ".dat";
                    DataModelIF<Long, Long> evalModel2 = new SimpleParser().parseData(new File(r2));
                    System.out.println(r1 + " vs " + r2);

                    AbstractRankingMetric<Long, Long> m1 = new NDCG<>(evalModel1, test, 5.0, cutoffs, NDCG.TYPE.EXP);
                    AbstractRankingMetric<Long, Long> m2 = new NDCG<>(evalModel2, test, 5.0, cutoffs, NDCG.TYPE.EXP);
                    m1.compute();
                    m2.compute();
                    for (int cutoff : cutoffs) {
                        Map<Long, Double> results1 = new HashMap<>();
                        for (Long u : m1.getValuePerUser().keySet()) {
                            results1.put(u, m1.getValueAt(u, cutoff));
                        }
                        Map<Long, Double> results2 = new HashMap<>();
                        for (Long u : m2.getValuePerUser().keySet()) {
                            results2.put(u, m2.getValueAt(u, cutoff));
                        }
                        for (String method : new String[]{"pairedT", "wilcoxon"}) {
                            double p = new StatisticalSignificance(results1, results2).getPValue(method);
                            System.out.println("\t" + m1 + "@" + cutoff + " stat.sign. using " + method + " with p=" + p);
                        }
                        double es = new EffectSize<>(results1, results2).getEffectSize("pairedT");
                        // **** Note: there are actually more methods available to compute the effect size...
                        System.out.println("\t" + m1 + "@" + cutoff + " effect size=" + es);
                    }
                }
            }
        }
    }

    private static Properties getLenskitProperties(String method) {
        Properties p = new Properties();
        p.put(RecommendationRunner.FRAMEWORK, "lenskit");
        switch (method) {
            case "svd50":
                p.put(RecommendationRunner.FACTORS, "50");
                p.put(RecommendationRunner.ITERATIONS, "10");
                p.put(RecommendationRunner.RECOMMENDER, "org.lenskit.mf.funksvd.FunkSVDItemScorer");
                break;

            case "ubknn50":
                p.put(RecommendationRunner.NEIGHBORHOOD, "50");
                p.put(RecommendationRunner.RECOMMENDER, "org.lenskit.knn.user.UserUserItemScorer");
                p.put(RecommendationRunner.SIMILARITY, "org.lenskit.similarity.CosineVectorSimilarity");
                break;
            default:
                throw new AssertionError();
        }
        return p;
    }

    private static Properties getRanksysProperties(String method) {
        Properties p = new Properties();
        p.put(RecommendationRunner.FRAMEWORK, "ranksys");
        switch (method) {
            case "svd50":
                p.put(RecommendationRunner.FACTORS, "50");
                p.put(RecommendationRunner.ITERATIONS, "10");
                p.put(RecommendationRunner.RECOMMENDER, "es.uam.eps.ir.ranksys.mf.rec.MFRecommender");
                p.put(RecommendationRunner.FACTORIZER, "es.uam.eps.ir.ranksys.mf.als.HKVFactorizer");
                break;

            case "ubknn50":
                p.put(RecommendationRunner.NEIGHBORHOOD, "50");
                p.put(RecommendationRunner.RECOMMENDER, "es.uam.eps.ir.ranksys.nn.user.UserNeighborhoodRecommender");
                p.put(RecommendationRunner.SIMILARITY, "es.uam.eps.ir.ranksys.nn.user.sim.VectorCosineUserSimilarity");
                break;
            default:
                throw new AssertionError();
        }
        return p;
    }

    private static Properties getMahoutProperties(String method) {
        Properties p = new Properties();
        p.put(RecommendationRunner.FRAMEWORK, "mahout");
        switch (method) {
            case "svd50":
                p.put(RecommendationRunner.FACTORS, "50");
                p.put(RecommendationRunner.ITERATIONS, "10");
                p.put(RecommendationRunner.RECOMMENDER, "org.apache.mahout.cf.taste.impl.recommender.svd.SVDRecommender");
                p.put(RecommendationRunner.FACTORIZER, "org.apache.mahout.cf.taste.impl.recommender.svd.SVDPlusPlusFactorizer");
                break;

            case "ubknn50":
                p.put(RecommendationRunner.NEIGHBORHOOD, "50");
                p.put(RecommendationRunner.RECOMMENDER, "org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender");
                p.put(RecommendationRunner.SIMILARITY, "org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity");
                break;
            default:
                throw new AssertionError();
        }
        return p;
    }
}
