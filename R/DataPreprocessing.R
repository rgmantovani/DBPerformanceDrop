# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

DataPreprocessing = function(data, define.targets = FALSE) {
	
	# converting characters to factors 
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
	
	# define Labels
	if(define.targets) {
		data5 = DefineTargets(temp = data5)
	}

	# all atributtes must be numeric, except the target
	for(i in 1:(ncol(data5)-1)) {
		data5[,i] = as.numeric(data5[,i])
	}

	# Scale data between [0,1]
	data6 = mlr::normalizeFeatures(obj = data5, target = "Target", method = "range", range = c(0,1))
	return(data6)
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

# df4 = Se (Uso de CPU >= 75%) OU (Memoria >= 95%) OU (CPU_LOAD_SHORT >=5 ) Então QUEDA. 
# if df.loc[i.Index, 'CPU_LOAD_SHORT'] == 1 or df.loc[i.Index, 'USO_MEMORIA'] == 1 or 
# int(df.loc[i.Index, 'TAMANHO_DO_BANCO_GB']) <= 25: # OK!

DefineTargets = function(temp) {
	
	temp$Target = 0
	
	# checking instances where the use of CPU is above 75%
	op1.ids = which(temp$USO_TOTAL_CPU >= 75) 

	# checking instances where the use of memory is above 95%
	memoryUse = temp$MEMORIA_FREE/temp$MEMORIA_TOTAL
	op2.ids = which(memoryUse >= 0.95) 

	# checking instances where the use of CPU load short is above 5%
	op3.ids = which(temp$CPU_LOAD_SHORT > 5)
	
	# unifying all the identified ids
	ids = union(union(op1.ids, op2.ids), op3.ids)
	temp[ids, "Target"] = 1

	# removing features used to define Target
	temp$MEMORIA_FREE   = NULL 
	temp$MEMORIA_TOTAL  = NULL
	temp$USO_TOTAL_CPU  = NULL
	temp$CPU_LOAD_SHORT = NULL

	temp$Target = as.factor(temp$Target)

	return(temp)
}

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------