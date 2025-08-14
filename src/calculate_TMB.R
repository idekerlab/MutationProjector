# calculate TMB from MAF files
library(maftools)
setwd('C:/Users/j4kong/Desktop/research/GPAcell/dataset/Project_Genie_cBioPortal/')
output_fldr = 'C:/Users/j4kong/Desktop/research/GPAcell/dataset/Project_Genie_cBioPortal/TMB/'

# read and calculate tmb
maf.file = read.maf('./data_mutations_extended.txt', verbose=TRUE)
calc_tmb = tmb(maf=maf.file)
# output dataframe
write.csv(calc_tmb, paste0(output_fldr, 'TMB.csv'), row.names=FALSE, quote=FALSE)

