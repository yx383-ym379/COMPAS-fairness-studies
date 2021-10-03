# Project Proposal
## Problem Statement
We recognize that a model like the COMPAS algorithm can be a “Weapon of Math Destruction”, as it satisfies the definitions: 1) its prediction outcome is not easily measurable, 2) false predictions of the model can cause serious trouble (reoffending or unnecessarily keeping the defendant in jail), 3) false predictions, especially false positive prediction becomes a negative feedback loop. 

Thus, we are interested in improving algorithmic fairness in recidivism algorithms like COMPAS by answering the following questions:
- Define and measure the fairness of a classification process/outcome.
- Optimize the performance of an algorithm while satisfying the fairness constraints. 
- Explore and quantify the implications of the algorithm to public safety. 
 
## Description of the dataset:
[Data Source](https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis)

Correctional Offender Management Profiling for Alternative Sanction(Compas) has been using its predictive algorithm to forecast an inmate’s likelihood of recidivism within two years after release from prison. Propublica, the data source, used the data from the Compas risk assessment system and questioned the basis of the Compas scores in 2016, stating that the unfairness was rooted in the algorithm and generated biased predictions not in favor of African American Felonies. The dataset offered by Propublica contains 7214 rows of unique recent inmates’ information from Broward County, Florida in 2013 and 2014. We understand that some basic cleaning is required for the dataset to filter out individuals that don’t perfectly fit the description of “recent inmates”. In total there are 28 columns of features which can be categorized as:
- Personal Attributes(Race, Gender, Marital Status)
- Previous Arrest Record
- Current Arrest Status 
- Target Variable (Decile score)

It’s worth noting that the sensitive attributes, which are commonly believed to result in the unfairness in the Compas predictions, are race and gender. We are curious if there are other candidates for the sensitive attributes.  And the target variable is a score given to each inmate to predict their likelihood of reentering the jail system within 2 years(A benchmark set by the Compas algorithm). Due to its popularity and controversy in the arena of fairness analysis in ML, we are interested in diving deeper into this dataset and see how we can improve our definition of fairness while achieving high predictive accuracy for future criminal predictions, standing on the shoulders of our peers. 


# Appendix:
## Literature review
The common understanding is that to satisfy fairness constraints, the classifier must sacrifice its performance in terms of accuracy. Many previous studies have been conducted in the search for the most suitable approach to optimize fairness with an acceptable model accuracy tradeoff. The different approaches can be divided into three categories by their mechanism: pre-process, in-process, and post-process. Among the various solutions, we are most interested in exploring the performance of model-agnostic approaches. As they can be easily adapted to different recidivism algorithms, which might or might not be publicly available.

In [Optimized Pre-Processing for Discrimination Prevention](https://papers.nips.cc/paper/2017/file/9a49a25d845a483fae4be7e341368e36-Paper.pdf), the authors suggest that discrimination prevention during data preprocessing stage of ML is preferable as it is model-agnostic. They demonstrate that fairness, measured by non-discrimination, can be achieved through preprocessing the training dataset. Another model-agnostic approach is discussed in [Addressing Fairness in Classification with a Model-Agnostic Multi-Objective Algorithm](https://www.auai.org/uai2021/pdf/uai2021.232.pdf), in which the optimization for fairness is done “in-process”, or during the training stage. [Minimax Pareto Fairness: A Multi Objective Perspective](http://proceedings.mlr.press/v119/martinez20a/martinez20a.pdf) shares the idea of viewing fairness-accuracy optimization as a multi-objective optimization problem, and proposes a novel fairness metric called Minimax Pareto Fairness. The authors of [Algorithmic decision making and the cost of fairness](https://arxiv.org/abs/1701.08230) propose a post-process approach of debiasing and introduce the idea of public safety risks as a measurement of model performance. 
