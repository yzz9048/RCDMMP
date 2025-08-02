# RCDMMP

# For now, only the core code is displayed and the complete code is coming soon.

# Datasets
For AIops18, please refer to https://github.com/BEbillionaireUSD/Maat

For GAIA, please refer to https://github.com/CloudWise-OpenSource/GAIA-DataSet

Media and SN are coming soon, and we present a toy dataset (the illustration expriment in the paper), similar to Media and SN, except that the data volume is smaller: https://drive.google.com/file/d/1sBylsgSY8jP1WHHCtipWOKgF8nCHoL1u/view?usp=drive_link

# Details
The json files in the data warehouse are the raw data collected by tools, i.e.., Prometheus and k6, and the csv files are extracted from json files. 
Metrics are collected at intervals of 10 seconds. Therefore, RPS data can perform average aggregation at intervals of 10 seconds as guiding condition data. We have provided the specific number of requests per second. Readers can handle and generate other conditional data by themselves.

This toy dataset is collected from microservice benchmark SocialNetwork, which comprises 13 microservices and 13 database microservices, each microservice has four instances deployed on two worker servers, each equipped with Intel(R) Xeon(R) CPU E5-2620 v3 @ 2.40GHz, 4* 8G DDR4 Registered (Buffered) 2133 MHz, 2T HHD. A master server (the same equipment as work server) generates workloads using k6, collects monitoring data via Prometheus.

File_a.py is the the core code of the diffusion model in the paper, and the File_b.py is the backbone of the diffusion model.

#For Media and SN

There are two folders in the dataset warehouse, namely Media and SN(Coming soon): 

the workload script for Media can be seen in compose-review.js, and mix-k6.js for SN

There is no failure contained in Media, but some failures are injected in SN, which has been prepared for further study, and the main type of failue injected is cpu failure, using "blade create k8s container-cpu load --cpu-percent xx --container-ids xx --names xx --kubeconfig xx --namespace xx". The last column of the table that combines RPS and metrics is the failure label.
