# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

# ------------------------------
# Installing required packages
# ------------------------------

# If they are not installed, please, uncomment the following lines:
# install.packages(c("ranger", "e1071", kknn", "mlr", "rpart", 
#   "dplyr", "reshape2", "ggplot2"))

# ---------------------------
# Loading packages
# ---------------------------

library(mlr, quietly = TRUE, warn.conflicts = FALSE)

# ---------------------------
# Seed for reproducibility
# ---------------------------

set.seed(123)

# ---------------------------
# MLR settings
# ---------------------------

mlr::configureMlr(on.learner.error = "warn")
mlr::configureMlr(show.info = TRUE)

# ------------------------------
# Feature names
# ------------------------------

# features selected by an expert
EXPERT.FEATURES = c("DB_NAME","CPU_LOAD_SHORT","USO_TOTAL_CPU","NR_CPU","MEMORIA_TOTAL",
 "MEMORIA_FREE","QTD_CONEXAO","BUILD_BANCO_DB2","BUILD_INST_DB2","QTD_ERROS_ATUALIZACAO",
 "ULTIMO_RUNSTATS","QTD_OBJETOS_INVALIDOS","PACOTES_INVALIDOS",
 "BUFFERPOLLS_AUTO","MEMORIA_BD","DFT_QUERYOPT","LOGBUFSZ","DB_MEM_THRESH","SELF_TUNING_MEM",
 "DATABASE_MEMORY","DFT_DEGREE","TAMANHO_DO_BANCO_GB","TRIGGERS_AUDLOG","INSTANCE_MEMORY")
# DTHORACONSULTA, DT_PACOTE_ANTIGO

DROP.FEATURES = c("HITRATIO_COL1", "HITRATIO_COL2", "HITRATIO_COL3", "BKP_LOCATION",
    "DTHORACONSULTA","DT_PACOTE_ANTIGO") # integrate dates after

# ------------------------------
# ------------------------------

ALGOS.MLR  = c("rpart", "kknn", "multinom", "ranger", "svm", "nnet", "naiveBayes")
   
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
