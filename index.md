---
layout: default
---

This is a supplementary website for the tutorial on **Replication of Recommender Systems Research** held at the [ACM RecSys Summer School](http://pro.unibz.it/projects/schoolrecsys17/).  

# [](#preparation)Preparation
Before attending the tuturial, make sure to prepare the following:

1. Download and familiarize yourself with [Lenskit](http://www.lenskit.org), [Librec](http://www.librec.net), or [RankSys](http://www.ranksys.org).
1. Download and extract the [Movielens 100k](http://files.grouplens.org/datasets/movielens/ml-latest-small-README.html) dataset.

During the hand-on session, you will be expected to be able to execute and evaluate a recommendation using your selected tool.

# [](#slides)Slidedeck
The slidedeck from the tutoral will be available [here](recsys-replication.pdf) before the tutorial takes place.

# [](#code) Code examples
Check out the following github repository [rsss2017](https://github.com/recommenders/rsss2017).
The repository contains a Maven project with two classes **ControlledEvaluation.java** and **RankSysEvaluation.java**. 

1. **ControlledEvaluation** performs a transparent, standalone, evaluation of RankSys, LensKit, and Mahout using [RiVal](http://rival.recommenders.net). The evaluation metrics presented are fully comparable as the data splitting, candidate item generation, and metric calculation is done identically for all three cases.
1. **RankSysEvaluation** performs the same type of evaluation, using RankSys's internal evaluation methods. The evaluation metrics are only comparable with other evaluation made within RankSys with the same settings.

# [](#instructors)Instructors
The tutorial is given by [Alejandro Bellogín](http://ir.ii.uam.es/~alejandro/) and [Alan Said](http://www.alansaid.com).