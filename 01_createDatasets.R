# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

set.seed(42)

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
# Create datasets
# -------------------------------------------------------------------------------------------------

# defineLabels
 
 = DefineLabels(data = , method = "cpu") 
 # = DefineLabels(data = , method = "memory")
 # = DefineLabels(data = , method = "cpu_load")
 # = DefineLabels(data = , method = "mixed")

# consumo de CPU >=75%, então queda (1), senão de boa (0)
# df1 = data 
# df1$Label = 0
# which(df1$USO_TOTAL_CPU)

# USO_TOTAL_CPU


# memoria_free/memoria_total
# consumo de memória maior que 95 (%), consideramos ruim 
#se memoria do sistema em uso > 95%, então queda (1), senao de boa (0)
# df2 = 


# se CPU_LOAD_SHORT > 5, então queda (1), senão de boa (0)
# df3 = 


# df4 = Se (Uso de CPU >= 75%) OU (Memoria >= 95%) OU (CPU_LOAD_SHORT >=5 ) Então QUEDA. 
 # if df.loc[i.Index, 'CPU_LOAD_SHORT'] == 1 or df.loc[i.Index, 'USO_MEMORIA'] == 1 or int(df.loc[i.Index, 'TAMANHO_DO_BANCO_GB']) <= 25: # OK!


# df[['DB_NAME','DTHORACONSULTA','CPU_LOAD_SHORT','USO_TOTAL_CPU','NR_CPU','MEMORIA_TOTAL',
# 'MEMORIA_FREE','QTD_CONEXAO','BUILD_BANCO_DB2','BUILD_INST_DB2','QTD_ERROS_ATUALIZACAO',
# 'ULTIMO_RUNSTATS','QTD_OBJETOS_INVALIDOS','DT_PACOTE_ANTIGO','PACOTES_INVALIDOS',
# 'BUFFERPOLLS_AUTO','MEMORIA_BD','DFT_QUERYOPT','LOGBUFSZ','DB_MEM_THRESH','SELF_TUNING_MEM',
# 'DATABASE_MEMORY','DFT_DEGREE','TAMANHO_DO_BANCO_GB','TRIGGERS_AUDLOG','INSTANCE_MEMORY']]



# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------