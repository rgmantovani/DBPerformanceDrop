# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

set.seed(42)

# TODO: list all R files in a list
# load all files

source("./R/config.R")
source("./R/DataPreprocessing.R")

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

data.original = read.csv("./data/originalData.csv")
data.original = data.original[ ,-which(colnames(data.original) %in% DROP.FEATURES)]

# for(cn in colnames(data.original)) {
# 	cat("* ", cn, " = ", class(data.original[,cn]), "\n")
# }

# converting some features to the correct type
data.original$CPU_LOAD_SHORT = as.numeric(data.original$CPU_LOAD_SHORT)
data.original$TAMANHO_DO_BANCO_GB = as.numeric(gsub(x = data.original$TAMANHO_DO_BANCO_GB, 
	pattern = ",", replacement = "."))

data.filtered = data.original[, EXPERT.FEATURES]

# > dim(data.original)
# [1] 5684  100
# > dim(data.filter)
# [1] 5684   24

# -------------------------------------------------------------------------------------------------
# Preprocessing datasets
# -------------------------------------------------------------------------------------------------

data.original.preproc = DataPreprocessing(data = data.original)
# > dim(data.preproc)
# [1] 5684   52

data.filtered.preproc = DataPreprocessing(data = data.filtered)
# dim(data.filtered.preproc)
# 5684   21

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------