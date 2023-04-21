# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

DataPreprocessing = function(data) {
	
	#character to factor 
	data1 = data 
	for(cn in colnames(data1)) {
		if(class(data[,cn]) == "character") {
			data1[,cn] = as.factor(data1[,cn])
		}
	}

	# removing ids and NA features
	data2 = mlr::removeConstantFeatures(obj = data1 , perc = 0.05, na.ignore = FALSE)

	# data imputation
	res = mlr::impute(
	  	obj = data2,
	  	classes = list(
			factor  = mlr::imputeConstant("NewValue"),
			integer = mlr::imputeMean(),
			logical = mlr::imputeMode() 
	  	)
	)

	data3 = res$data

	# TODO: remove high correlated features
	
	return(data3)
}

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

DataPreProcessingNumerical = function(data) {
	# categoricos -> numerics/binários
	#obj = mlr::createDummyFeatures(obj = data3, method = "1-of-n")
	# normalização entre [0,1]
}
	
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
