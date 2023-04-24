# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

DefineLabel = function(temp) {
	
	temp$Target = 0
	
	# checking instances where the use of CPU is above 75%
	op1.ids = which(temp$USO_TOTAL_CPU >= 75) 

	# checking instances where the use of memory is above 95%
	memoryUse = temp$MEMORIA_FREE/temp$MEMORIA_TOTAL
	op2.ids = which(memoryUse >= 0.95) 

	# checking instances where the use of CPU load short is above 5%
	op3.ids = which(temp$CPU_LOAD_SHORT > 5)
	
	# unifying all the identified ids
	ids = union(op1.ids, op2.ids, op3.ids)
	temp[ids, "Target"] = 1

	# removing features used to define Target
	temp$MEMORIA_FREE  = NULL 
	temp$MEMORIA_TOTAL = NULL
	temp$USO_TOTAL_CPU = NULL
	temp$CPU_LOAD_SHORT = NULL

	temp$Target = as.factor(temp$Target)

	return(temp)
}

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
