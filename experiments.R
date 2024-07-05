require("shinystan")
require("rstan")
require("purrr")

files <- purrr::map_vec(list.files("csv"), function (s) paste("csv/", s, sep=""))
stanfit <- rstan::read_stan_csv(files)
shinystan::launch_shinystan(stanfit)
