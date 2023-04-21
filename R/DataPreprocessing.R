# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

DataPreprocessing = function(data) {
	
	#character to factor 
	data1 = data 
	for(cn in colnames(data1)) {
		if(class(data1[,cn]) == "character") {
			data1[,cn] = as.factor(data1[,cn])
			
		} 
		na.ids = which(data1[,cn] == "")
		data1[na.ids, cn] = NA	
	}

	# removing ids and NA features
	data2 = mlr::removeConstantFeatures(obj = data1 , perc = 0.05, na.ignore = FALSE)

	# data imputation
	res = mlr::impute(
	  	obj = data2,
	  	classes = list(
			factor  = mlr::imputeConstant("NewValue"),
			integer = mlr::imputeMean(),
			numeric = mlr::imputeMean(),
			logical = mlr::imputeMode() 
	  	)
	)
	data3 = res$data

	# removing constant featrues after imputation
	data4 = mlr::removeConstantFeatures(obj = data3 , perc = 0.05, na.ignore = FALSE)

	# removing high correlated features
	data5 = RemoveHighCorrelatedFeatures(data = data4, perc.cor = 0.9)
	
	return(data5)
}

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

RemoveHighCorrelatedFeatures = function(data, perc.cor = 0.99) {
	
	temp = data
	for(cn in colnames(data)) {
		if(class(data[, cn]) != "numeric") {
			temp[,cn] = as.numeric(data[, cn])
		}
	}

	corMatrix = cor(temp)

	# Modify correlation matrix
	corMatrixRM = corMatrix                  
	corMatrixRM[upper.tri(corMatrixRM)] = 0
	diag(corMatrixRM) = 0

	# removing high correlated features
	dataNew = data[ , !apply(corMatrixRM, 2, function(x) any(x > perc.cor))]
	return(dataNew)
}


# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

# DataPreProcessingNumerical = function(data) {
# 	# categoricos -> numerics/binários
# 	#obj = mlr::createDummyFeatures(obj = data3, method = "1-of-n")
# 	# normalização entre [0,1]
# }
	
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
