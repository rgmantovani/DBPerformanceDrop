# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

set.seed(42)

R.files = list.files(path="./R", full.names=TRUE)
for(ffile in R.files) {
	source(ffile)
}

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

cat(" @Loading files\n")

data.original = read.csv("./data/originalData.csv")
data.original = data.original[ ,-which(colnames(data.original) %in% DROP.FEATURES)]

# converting some features to the correct type
data.original$CPU_LOAD_SHORT = as.numeric(data.original$CPU_LOAD_SHORT)
data.original$TAMANHO_DO_BANCO_GB = as.numeric(gsub(x = data.original$TAMANHO_DO_BANCO_GB, 
	pattern = ",", replacement = "."))

cat(" @Filtering features based on the specialist\n")
data.filtered = data.original[, EXPERT.FEATURES]
data.filtered = DefineTargets(temp = data.filtered)

# -------------------------------------------------------------------------------------------------
# Preprocessing datasets
# -------------------------------------------------------------------------------------------------

cat(" @Preprocessing datasets\n")

cat("* Before preprocessing: \n")
cat(" - Original dataset dimensions: ", dim(data.original), "\n")
cat(" - Specialist dataset dimensions: ", dim(data.filtered), "\n")

data.original.preproc = DataPreprocessing(data = data.original, define.targets = TRUE)
data.filtered.preproc = DataPreprocessing(data = data.filtered, define.targets = FALSE)

cat("* After preprocessing: \n")
cat(" - Automated dataset dimensions: ", dim(data.original.preproc), "\n")
cat(" - Specialist dataset dimensions: ", dim(data.filtered.preproc), "\n")

# -------------------------------------------------------------------------------------------------
# Exporting data files
# -------------------------------------------------------------------------------------------------

cat(" @Exporting data files\n")
write.csv(data.filtered, file = "data/dataset_specialist.csv")
write.csv(data.filtered.preproc, file = "data/dataset_specialist_preprocessed.csv")
write.csv(data.original.preproc, file = "data/dataset_original_preprocessed.csv")

cat("Done!\n")

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------