#Rscript ../scripts/get_metadata_slendr.R /maps/projects/racimolab/people/qxz396/spaceNNtime/data/europe 1e-8 1e-8 3000 1234
reticulate::use_condaenv("sNNt_ts")
library(ggplot2)
library(cowplot)
library(slendr)
library(sf)
library(tidyverse)


options  <- commandArgs(trailingOnly = TRUE)
sim_path <- options[1] #"/maps/projects/racimolab/people/qxz396/spaceNNtime/data/europe"
rec_rate <- as.numeric(options[2]) #1e-8
mut_rate <- as.numeric(options[3]) #1e-8
ne       <- as.numeric(options[4]) #3000
ran_seed <- as.numeric(options[5]) #1234

model <- read_model(sim_path)
ts    <- ts_load(model, recapitate = TRUE, simplify = TRUE, 
                 recombination_rate = rec_rate, Ne = ne, random_seed = ran_seed) %>%
            ts_mutate(., mutation_rate = mut_rate, random_seed = ran_seed)

ts_save(ts, file = paste(sim_path, "/tree.trees", sep = ""))

data  <- ts_data(ts)

data.frame(name       = data$name,        ind_id  = data$ind_id,  pedigree_id = data$pedigree_id, 
           pop        = data$pop,         node_id = data$node_id, location    = data$location, 
           time       = data$time,        sampled = data$sampled) %>%
    filter(sampled) %>%
    rowwise() %>% 
    mutate(lat = st_transform(geometry, 4326)[[1]][1], 
           lon = st_transform(geometry, 4326)[[1]][2]) %>% 
           select(-c(geometry, name, pedigree_id)) %>%
    group_by(ind_id, pop, time, sampled, lat, lon) %>%
    mutate(n = 1, n = cumsum(n), n = paste("node", n, sep = "")) %>%
    spread(n, node_id) %>%
    ungroup() %>%
    write.table(., file = paste(sim_path, "/metadata.txt", sep = ""), 
                append = FALSE, quote = FALSE, sep = "\t", row.names = FALSE, col.names = TRUE)