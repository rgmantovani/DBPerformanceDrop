## ------------------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------------------

cat(" @ Loading all required files:\n")
R.files = list.files(path = "R", full.names = TRUE)
for(file in R.files) {
  source(file)
}

# TODO: if results are available, does not run again, skip to analysis

## ------------------------------------------------------------------------------------------
## 1. Reading data
## ------------------------------------------------------------------------------------------

cat(" @ Loading Dataset\n")

data1 = read.csv("data/dataset_original_preprocessed.csv")[,-1]
data2 = read.csv("data/dataset_specialist.csv")[,-1]
data3 = read.csv("data/dataset_specialist_preprocessed.csv")[,-1]

## ------------------------------------------------------------------------------------------
## 2. Creating output dirs 
## ------------------------------------------------------------------------------------------

cat(" @ Creating output folders\n")
if(!dir.exists("results/")) {
  dir.create("results/", showWarnings=FALSE, recursive=TRUE)
}

## ------------------------------------------------------------------------------------------
## 3. ML training using mlr3
## ------------------------------------------------------------------------------------------

cat("@ Creating learning tasks\n")
# a) creating learning tasks (classification)
task1 = mlr::makeClassifTask(id = "Original_Preprocessed", data = data1, target = "Target")
task2 = mlr::makeClassifTask(id = "Specialist", data = data2, target = "Target")
task3 = mlr::makeClassifTask(id = "Specialist_Preprocessed", data = data3, target = "Target")

## ------------------------------------------------------------------------------------------
#  list of all tasks
tasks = list(task_cor, task_all, task_esta, task_rest, task_all2)
print(tasks)

## ------------------------------------------------------------------------------------------
# b) creating the learning algorithms (using an inner function)

cat("@ Creating ML learners\n")
learners     = createLearnersMLR()
print(learners)

## ------------------------------------------------------------------------------------------
# c) choosing evaluation measures
# BAC  : Balanced Accuracy per Class
# F1   : FScore
# GMean: Geometrical Mean

# measures = list(mlr::bac, f1_multiclass_measure, gmean_multiclass_measure)
# print(measures)

## ------------------------------------------------------------------------------------------
# d) Define a resampling 

# for debug purposes
# cv = mlr::makeResampleDesc(method = "RepCV", folds = 10, rep = 3, stratify = TRUE)
# cv = mlr::makeResampleDesc(method = "RepCV", folds = 10, rep = 10, stratify = TRUE)
# print(cv)

## ------------------------------------------------------------------------------------------
# e) execute experiments (tasks, algorithms) -> benchmark

cat("@ Running experiment\n")
# res = mlr::benchmark(learners = learners, tasks = tasks, resamplings = cv, 
    # measures = measures, show.info = TRUE, keep.pred = TRUE, models = FALSE)
# print(res)

# complete results
# perf.complete   = mlr::getBMRPerformances(bmr = res, as.df = TRUE)
# perf.aggregated = mlr::getBMRAggrPerformances(bmr = res, as.df = TRUE)

cat("@ Saving results\n")
# use load() to read it again
# save(res, file = "results/mlr_results_complete.RData")
# write.csv(perf.complete, file = "results/mlr_performances_complete.csv")
# write.csv(perf.aggregated, file = "results/mlr_performances_aggregated.csv")

cat("Done !!! \n")

## ------------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------------
