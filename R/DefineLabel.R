# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

DefineLabel = function(temp, method) {
	
	temp$Target = 0
	
	op1.ids = which(temp$USO_TOTAL_CPU >= 75) 

	memoryUse = temp$MEMORIA_FREE/temp$MEMORIA_TOTAL
	op2.ids = which(memoryUse >= 0.95) 

	op3.ids = which(temp$CPU_LOAD_SHORT > 5)
	 
	ids = union(op1.ids, op2.ids, op3.ids)

	temp$MEMORIA_FREE  = NULL 
	temp$MEMORIA_TOTAL = NULL
	temp$USO_TOTAL_CPU = NULL
	temp$CPU_LOAD_SHORT = NULL

	temp[ids, "Target"] = 1

	temp$Target = as.factor(temp$Target)

	return(temp)
}

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
