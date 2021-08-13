my_data <- read.csv("path.csv", header=T)

latency_model <- lm(formula = my_data$latency ~ 
                      my_data$tot_24+my_data$tot_40+my_data$tot_80+my_data$tot_112+my_data$tot_160
                      
                    +my_data$avg_e_24+my_data$avg_e_40+my_data$avg_e_80+my_data$avg_e_112+my_data$avg_e_160
                    +my_data$avg_k_24+my_data$avg_k_40+my_data$avg_k_80+my_data$avg_k_112+my_data$avg_k_160
                    
                    +my_data$tot_e_16_24+my_data$tot_e_24_40+my_data$tot_e_40_80+my_data$tot_e_80_112+my_data$tot_e_112_160
                    +my_data$tot_k_16_24+my_data$tot_k_24_40+my_data$tot_k_40_80+my_data$tot_k_80_112+my_data$tot_k_112_160
                    
                    +my_data$avg_e_24*my_data$tot_24
                    +my_data$avg_e_40*my_data$tot_40
                    +my_data$avg_e_80*my_data$tot_80
                    +my_data$avg_e_112*my_data$tot_112
                    +my_data$avg_e_160*my_data$tot_160
                    
                    +my_data$avg_k_24*my_data$tot_24
                    +my_data$avg_k_40*my_data$tot_40
                    +my_data$avg_k_80*my_data$tot_80
                    +my_data$avg_k_112*my_data$tot_112
                    +my_data$avg_k_160*my_data$tot_160
)

summary(latency_model)
latency_model
plot(latency_model)

