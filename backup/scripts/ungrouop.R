library(tidyverse)

args <- commandArgs(trailingOnly = TRUE)
exp <- args[1]
cro <- args[2]
sta <- args[3]
end <- args[4]
win <- args[5]


read.table(paste("/home/moicoll/spaceNNtime/sandbox/AADR/", exp, "/pred_", cro, "_", sta, "_", end, "_", win, ".txt", sep = ""), header = T) %>%
    select(sta) %>%
    distinct() %>%
    pull(sta) -> starts 

read.table(paste("/home/moicoll/spaceNNtime/sandbox/AADR/", exp, "/pred_", cro, "_", sta, "_", end, "_", win, ".txt", sep = ""), header = T) %>%
    select(end) %>%
    distinct() %>%
    pull(end) -> ends

for(i in 1:length(starts)){
    print(paste(starts[i], "to", ends[i]))
    read.table(paste("/home/moicoll/spaceNNtime/sandbox/AADR/", exp, "/pred_", cro, "_", sta, "_", end, "_", win, ".txt", sep = ""), header = T) %>%
        filter(sta == starts[i]) %>%
        write.table(., file = paste("/home/moicoll/spaceNNtime/sandbox/AADR/", exp, "/pred_", cro, "_", starts[i], "_", ends[i], ".txt", sep = ""), sep = "\t", quote = FALSE)
}