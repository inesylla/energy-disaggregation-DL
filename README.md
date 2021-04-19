# Non-intrusive Load Disaggregation 
## Introduction

- Motivation and goals

Climate change is one of the greatest challenges facing humanity, and machine learning approaches are a great solution to tackle this problem. In 2019, a group of machine learning experts developed a paper called "Tackling Climate Change with Machine Learning" [[1]](#1) focused on impactful uses of machine learning in reducing and responding to climate change challenges. 

One of the main domains of the many propositions is "Buildings and cities" and in more deep how to "optimize buildings energy consumption". The paper states "_while the energy consumed in buildings is responsible for a quarter of global energy-related emissions, a combination of easy-to-implement fixes and state-of-the-art strategies could reduce emissions for existing buildings by up to 90%_". This statement caught our attention to start this project. Find an optimization model to control and therefore optimize energy consumption in buildings. 

After extensive research, we decided to focus our study on Non-Intrusive Load Monitoring (NILM). NILM is the task of estimating the power demand of different appliances in a building given an aggregate power demand signal recorded by a single electric meter monitoring multiple appliances. 

Neural NILM is a non-linear regression problem that consists of training a neural network for each appliance in order to predict a time window of the appliance load given the corresponding time window of aggregated data.

We adopted the "Non-Intrusive Load Monitoring with an Attention-based Deep Neural Network" [[2]](#2) paper developed by University of Rome Tor Vergata researchers, to be our Reference Paper.  Other approaches to Neura NILM are presented in "Non-intrusive load disaggregation solutions for very low-rate smart meter data." [[3]](#3) and "Sequence-to-point learning with neural networks for non-intrusive load monitoring" [[4]](#4).

- Dataset

As to the dataset used, we selected the real-world dataset "the Reference Energy Disaggregation Data Set (REDD)" [[5]](#5). This dataset is one of the reference datasets used in NILM Reference Paper and contains data for six different houses from the USA. The data is collected at 1 second sampling period for the aggregate power consumption and 3 seconds for the appliance power consumption. The appliances used are the following: 
oven, refrigerator, dishwasher, kitchen_outlets, microwave, bathroom_outlet, lighting, washer_dryer, electric_heater, stove, disposal, electronics, furance, smoke_alarms, air_conditioner. 
Thus, in our model, we consider three appliances: dishwasher (DW), microwave (MW), and refrigerator (FR). These appliances are the same as the ones used in the Reference Paper to reach the same results.

Dataset split

The dataset is split using houses 2,3,4,5,6 to build the training set and house 1 as the test set. 

<img width="555" alt="Captura de pantalla 2021-04-16 a las 13 41 36" src="https://user-images.githubusercontent.com/71388892/115021656-88237f00-9ebc-11eb-85d4-6ecdbbb8ab08.png">

The actual dataset of our model is a combination of two datasets. 
- We found a deep learning research team from Seoul National University that had a pre-processed dataset that cleaned the data (see pre-processing section bellow) of the REDD dataset given by the Reference Paper. This dataset is used in their "Subtask Gated Networks for Non-Intrusive Load Monitoring paper" [[6]](#6).
- On the other hand, in the REDD dataset there is a high active/inactive windows imbalance. This irregularity is observed especially in the case of the dishwasher and the microwave. As it is expected, due to the use of these appliances, much of the time a dishwasher and a microwave are not being used. Therefore there is a high overrepresentation of inactive windows. We implemented an oversampling process described in the pre-processing section (see below) to solve the problem. 

## System architecture

### Preprocessing

Initial project implementation was done using raw REDD dataset and it was necessary to pre-process the data as described in "Subtask gated networks for non-intrusive load monitoring" [[6]](#6), see details:

1. Data alignment. Align multiple time series with different acquisition frequencies.
2. Data imputation. Split the sequence so that the duration of missing values in subsequence is less than 20 seconds. Then fill the missing values in each subsequence by a backward filling method.
3. Data filtering. Only use the subsequences with more than one day duration
4. Generate sliding windows. Using sliding window over the aggregated signal with hop size equal to 1 sample

Once authors from Seoul National University provided us the same dataset as the Reference Paper we disabled our data pre-processing. The main reason was to assure the same input data as the original paper to have the same, or similar, results.

Oversampling is used to solve the problem of overrepresentation of inactive windows and the irregulatity of the active/innactive windows imbalance (described in the Dataset section). The process consist in replicating randomly picked active windows in each of the appliances to obtain a 50% - 50% class balance. The ratio between active/inactive windows is configurable in settings. 

After implementing oversampling the number of windows used for train, eval and test are listed below:


| Appliance | Nº buildings train | Nº windows train | Nº windows eval | Nº buildings test |
|-----------|--------------------|------------------|-----------------|-------------------|
| dishwasher|                   5|          289163  |          123927 |                  1|
|   fridge  |                   4|          613167  |          262787 |                  1|
| microwave |                   3|           82922  |           35538 |                  1|


### Model architectures

We've implemented three different model architectures:

- Regression and classification enabled
- Only regression enabled.
- Regression and classification using the attention results.

![image](https://user-images.githubusercontent.com/7881377/115058996-3394f980-9ee6-11eb-874f-92aceee6f2f1.png)


#### Regression and classification enabled
The designed architecture adopted to solve the NILM problem is based on a classical end-to-end regression network with its encoder-decoder components. Adding an attention mechanism in between the encoder and decoder. Apart from the main end-to-end regression network, an auxiliary end-to-end classification subnetwork is joined.

Why an attention-based model? 
The attention-based model helps with the energy disaggregation task. It assigns importance, thought weights, to every position in the aggregated signal which after successful training, will correspond to a state change of the target appliance. The addition of an attention mechanism in the regression subnetwork will allow the model to focus on selected time steps or windows rather than on non-target appliances. 
The attention scores are the way to weigh the importance of every position in our input sequence to infer the disaggregated signal. To represent correctly these weights we made the output of the attention layer be a 1D vector with the length of a window sequence. 

Both subnetworks have a different objective: 
-	Regression end-to-end network: allows the subnetwork to “implicitly detect and assign more importance to some events (e.g. turning on or off of the appliance) and to specific signal sections”.
-	Classification end-to-end network: helps the disaggregation process by enforcing explicitly the on/off states of the appliances. 

Both subnetwork outcomes are concatenated at the end to outcome the disaggregated consumption of the appliances.

<img width="935" alt="Captura de pantalla 2021-04-16 a las 17 51 33" src="https://user-images.githubusercontent.com/71388892/115050909-8964a400-9edc-11eb-986f-41c5f83a7204.png">


#### Only regression enabled
This architecture consists of suppressing the classification subnetwork, that does not have an attention layer, from the model. The regression branch is kept as in the original network.

#### Regression and classification using the attention results
In this final model modification, the output of the attention layer is used to compute the result of the regression subnetwork (in all the models). In this architecture, we concatenate the output of the regression subnetwork with the output of the stack of convolutional layers, in the classification subnetwork. This concatenated vector is fed to the 2 fully connected layers on top of the classification branch. The expectations of this architecture's behavior are described in the Experiment 7 hypothesis.

### Train

- Methodology. Model training is done using the whole pre-processed train dataset and batches of size 64 via data loader. At first, we set the epochs at 10 epochs, in most of the cases we founded enough to do an initial analysis of model response and performance. The common do_load -> do_predict -> calculate_loss -> update_optimizer train sequence is done per each of the train batches in each epoch. The common do_load -> do_predict -> calculate_loss validation sequence is done per each of the validation batches in each epoch.

- Loss function. An aggregated loss function is used for the joint optimization of both regression and classification network: L=Lout+Lclas, where Lout is the Mean Squared Error (MSE) between the overall output of the network and the ground truth of a single appliance, and Lclas is the Binary Cross-Entropy (BCE) that measures the classification error of the on/off state for the classification subnetwork.


### Test

- Methodology. Model testing is done over the whole preprocessed test dataset using batches of size 64 via a data loader. The common do_load -> do_predict -> calculate_error test sequence is done per each of the test batches.
- Error metrics. MAE (Mean Absolute Error) is used to evaluate the performance of the neural network. MAE is calculated after applying the prediction postprocessing described in the Postprocessing section. These are the metrics used in the Reference paper and are used as benchmarking criteria between the different experiments described below.

### Postprocessing

The disaggregation phase is carried out with a sliding window over the aggregated signal with a hop size equal to 1 sample. That's the reason why the model generates overlapped windows of the disaggregated signal. We reconstruct the overlapped windows employing a median filter on the overlapped portion.

## Experiments

The main goals of the experiments are:

- Learn how to implement and deploy a DL system on a commercial cloud computing platform
- Understand and interpret the current NILM neural network described in the paper
   - Understand which is the task of regression branch
   - Understand which is the task of classification branch
   - Understand which is the task of attention

We proposed the three main architecture modifications evaluated in the experiments during the analysis of the reference paper. The experiments were not designed
sequentially after processing the results of the previous experiment. 

Main architecture modifications:

- Paper architecture - Regression and classification enabled
- Paper modification 1 - Only regression enabled
- Paper modification 2 - Regression and classification using the attention results

We initially explored the data to have a first picture of the type and the amount of data available. We realized there was a high active/inactive windows imbalance in the case of dishwasher and microwave (as explained in the Dataset explanation). There would be enough total amount of windows to train the model, but not enough specific active windows to prevent a biased model. If no oversample was done the model would mainly predict null demand in inactive windows, which would be correct, but would fail to predict non-null demand inactive windows. Although disaggregation is a regression problem, this would be similar to high specificity and low sensitivity in an active/inactive appliance classification problem. 

### Neural network response charts

We generate charts with time series describing the response of the neural network in train, eval, and test. These charts are used to visualize and interpret the response of both whole and specific parts of the network. The main parts of interest are regression, classification, and attention. In most of the charts, the available time series are:

- Building consumption. Aggregated consumption of the building. Used as input of the neural network
- Predicted appliance consumption. Disaggregated appliance consumption predicted by the neural network
- Real appliance consumption. Real applianced consumption obtained from the meter
- Classification branch output. Prediction of the classification branch
- Regression branch output. Prediction of the regression branch
- Attention score. Describes the zone of interest for attention to improve regression


![image](https://user-images.githubusercontent.com/7881377/115200957-141feb80-a0f5-11eb-82ca-81249810fe0b.png)


All the consumption time series are referenced to the left-Y axis. Classification and attention are referenced to the right-Y-axis. In both cases, there's a rescaling in some prediction results to make all of them fit in a single chart (ie. classification prediction is scaled to nearly maximum consumption, ...). In the report, there're two train and two test sample charts per each of the experiments and appliance to visualize the response and support conclusions.

Interpretation of the charts focuses in:
- Performance. Comparing real vs predicted series it's possible to identify the performance of the model
- Characterization of the error. Comparing real vs predicted series it's possible to identify error specific patterns (peaks, plateaus, etc)
- Correlation of the error with aggregated demand. Comparing error vs aggregated building consumption it's possible to identify the response of the model to crowded scenarios (multiple appliances) and single scenario (single appliance). It's also possible to identify the response of the model with different kinds of appliances, with different consumption patterns,  running simultaneously.
- Contribution of each of the branches. Analyzing the output of the branches is possible to identify the contribution of each of the branches to the prediction. It's possible to identify the objective of each branch and also its performance
- Focus of attention. Analyzing the attention output it's possible to identify which parts of the window are important to the regression output. The attention can be used to:
    - Identify whether the important parts vary in the different scenarios. Maybe there is a scenario in which there are different appliances ON or there is a scenario with just one appliance that is consuming a lot is being used ON. The attention will help differenciate this two situations. 
    - Identify whether there're specific important parts or the importance is homogeneous along with the window
    - Identify whether important parts are described in the appliance itself or the neighborhood.
    - Identify characteristic of important parts such as peaks, plateaus, etc.


## Paper architecture - Regression and classification enabled

### Experiment 1. Paper

#### Hypothesis

The regression subnetwork infers the power consumption, whereas the classification subnetwork focuses on the binary classification of the appliance state (on/off). The attention mechanism improves the representational power of the network to identify the positions in the aggregated input sequence with useful information to identify appliance-specific patterns. 

Additional group hypothesis:

Specific appliance patterns are described by state changes and state duration which are related to the operating regime of the internal electricity consumption components. The operating regime of the internal components depends on multiple factors:

- Appliance operating mode.
   - User-selected modes of operation. There're appliances with a small number of user modes (fridge, dishwasher) and appliances with a mid number of user modes (microwave). The higher number of user modes is the higher number of different patterns that can be described by the neural network.
   - Cycle duration. There're appliances with small duration time cycles describing the pattern per operating mode, such as the fridge and the microwave, and appliances with high duration time cycles, such as the dishwasher. The longer the cycle duration is, the more difficult it will be to describe the behavior of the pattern as the input sequence windows are longer.
- Environmental factors (temperature, etc). There're appliances with dependencies to external variables like environmental factors. In this specific model, there's a high dependency on temperature on the fridge and lower dependency on the microwave and dishwasher. Weather dependency adds stochasticity to the system and consequently,  complexity to the model.
- Internal components demand. The main electricity consuming components are:
  - Heating/cooling. There's weather dependency load demand adds stochasticity to the system, hence complexity to the model. 
  - Motors. Load demand is mainly related to the user mode and to the component internal operating regime.

#### Experiment setup

See details of the experiments below. Each of the columns describes a specific option of the previously introduced network architectures and pre/post-processing methods:

| Appliance | Regression | Classification | Standardization| Recalculate mean/std in test| 
|-----------|------------|----------------|----------------|-----------------------------|
| dishwasher|       TRUE |           TRUE |        FALSE   |                       FALSE |
| fridge    |       TRUE |           TRUE |        FALSE   |                       FALSE |
| microwave |       TRUE |           TRUE |        FALSE   |                       FALSE |

#### Results

See attached train vs loss curve to diagnose performance:

![image](https://user-images.githubusercontent.com/7881377/114513315-59967180-9c3a-11eb-974e-5e3d5eccacb8.png)

See attached train and test samples per each of the appliances to interpret end evaluate disaggregation:

![image](https://user-images.githubusercontent.com/7881377/114427700-a4bd6f80-9bbb-11eb-9936-c94c54ea2a0e.png)
![image](https://user-images.githubusercontent.com/7881377/114305400-58e5ca00-9ad8-11eb-87e0-08fe03fc2ed2.png)

<img width="259" alt="Captura de pantalla 2021-04-16 a las 20 25 23" src="https://user-images.githubusercontent.com/71388892/115068039-fc2c4a00-9ef1-11eb-80e8-cca07e338cf9.png">

See obtained error (previously introduced in error metrics section and extra training information):

| Appliance  | MAE   | Nº Epochs| Nº Hours Train|
|------------|-------|----------|---------------|
|dishwasher  |28.25  |4         |           15  |
|fridge      |26.75  |4         |           25  |
|microwave   |31.47  |4         |         1.23  |

#### Conclusions

As was described in the hypothesis the main goal of the regression branch is
predicting the maximum expected demand of the appliance.  As was also expected
the classification branch is modulating the regression results to match
the appliance load pattern. Classification has high specificity and low sensitivity.

In both cases, train and eval have good results but have
less accurate results in test. Our hypothesis is that
model does not generalize well due to the small number and variance of appliance patterns of the different train buildings.

See samples of dishwasher consumption per building:

![image](https://user-images.githubusercontent.com/7881377/114833109-862fc200-9dcf-11eb-8306-87bc1b769e80.png)

The classification network is in charge of modeling the patterns. As seen in the results, it is less accurate in the steady-state sections than expected. Hence, the instability, and in some cases, the high sensitive response is also related to the overrepresentation issue. 

In most cases, increasing the number of acquisition samples would not be a good solution to fix the instability issue as there would be more active windows but the same pattern. That's the case of appliances with components that do not depend on environmental factors (temperatures, etc) like microwave or dishwasher. In the case of appliances with environmental factors, it would help to have also samples from different seasons. We implemented oversampling but it's similar to increasing the number of samples from the same appliance rather than new ones. 

There's no more data available rather than the public dataset. As a solution, data augmentation can not be easily implemented due to the lack of a database of appliance loads. In this case, it makes no sense to create synthetic aggregated scenarios mixing appliances from different buildings because they're already mixed in the training dataset and properly predicted in eval. In the classification branch, we hypothesize that in some cases adding noise would help to decrease high sensitive responses.

Attention in appliances with a high simultaneity factor(\*) focus mainly on state changes in the appliance, like switch on/switch off or high consuming components of the appliance. Also, it focuses on state duration. That would be the case of dishwasher or
microwave. Attention in appliance with low
simultaneity factor also focus in other sections of the windows out of the
active section. That would be the case of the fridge. Our hypothesis is that in the case of high simultaneity factor
scenarios, attention focuses on appliance pattern, and in the case of low
simultaneity factors it additionally focuses on the neighborhood. Attention would perform better to identify highly specialized and specific features in a consumption window.

(\*) simultaneity factor describes the probability of an appliance
to be active while other appliances are active. A large simultaneity factor
means that the appliance is usually active while others are also active.
 
Regarding the hypothesis on type of appliances:
 - The neural network can model the different operating modes in the appliances, even the ones with a high number of operating modes
 - The neural network can model both heating/cooling and motor components
 - There's no specific conclusion about the capacity to model weather dependency as both train and test datasets were acquired under similar environments (season, etc)

### Experiment 2 and 3. Paper with standarization

Standardization can be used to rescale the testing samples to better
describe relative patterns rather than absolute value consumptions. Standardization transforms features such that their mean (μ) equals 0 and standard deviation (σ) equals 1. The range of the new min and max values is determined by the standard deviation of the initial un-normalized feature.

![Suource: https://becominghuman.ai/feature-scaling-in-machine-learning-20dd93bb1bcb](https://user-images.githubusercontent.com/7881377/115056599-57a30b80-9ee3-11eb-8aee-87f5e5c96401.png)

Standardization is achieved by Z-score Normalization. Z-score is given by:

![Source: https://becominghuman.ai/feature-scaling-in-machine-learning-20dd93bb1bcb](https://user-images.githubusercontent.com/7881377/115056913-aea8e080-9ee3-11eb-80bf-1e304274b3c5.png)

TThe standardization process is done over the specific dataset in each specific experiment.

Although the model of appliances in train and test are different in terms of absolute consumptions, relative step changes in standardized data can be similar.
This is an approach to bypass overrepresentation in data. In this case, the mean and standard value used in training is calculated over the train dataset, and the mean and standard deviation in the test is calculated over test dataset.

#### Experiment 2. Paper with standardization - Using calculated standardization in test

##### Hypothesis

The main difference between Experiment 1 and Experiment 2 is the addition of the standardization in the dataset of the model as explained in the past paragraph. In this experiment we are standarizing the data of the train in respect to the train and the test in respect to the test. 

<img width="401" alt="Captura de pantalla 2021-04-17 a las 12 49 29" src="https://user-images.githubusercontent.com/71388892/115110374-66d59800-9f7b-11eb-8cf6-a7d754aeeff8.png">

We hypothesize that we will have a better outcome than Experiment 1.

##### Experiment setup

See details of the experiments below. Each of the columns describes an specific option of the previously introduced network architectures and pre/post-processing methods:

| Appliance | Regression | Classification | Standardization| Recalculate mean/std in test| 
|-----------|------------|----------------|----------------|-----------------------------|
| dishwasher|       TRUE |           TRUE |          TRUE  |                       TRUE  |
| fridge    |       TRUE |           TRUE |          TRUE  |                       TRUE  |
| microwave |       TRUE |           TRUE |          TRUE  |                       TRUE  |

##### Results

See attached train and test samples per each of the appliances to interpret end evaluate disaggregation:

![image](https://user-images.githubusercontent.com/7881377/114307075-54241480-9ade-11eb-9063-4cdbcef3c081.png)
![image](https://user-images.githubusercontent.com/7881377/114307086-600fd680-9ade-11eb-921e-2803c2115931.png)
![image](https://user-images.githubusercontent.com/7881377/114307171-b67d1500-9ade-11eb-9b4a-ed3653b6c354.png)

<img width="259" alt="Captura de pantalla 2021-04-16 a las 20 25 23" src="https://user-images.githubusercontent.com/71388892/115068039-fc2c4a00-9ef1-11eb-80e8-cca07e338cf9.png">

See obtained error previously introduced in error metrics section and extra training information:

| Appliance  | MAE   | Nº Epochs| Nº Hours Train|
|------------|-------|----------|---------------|
|dishwasher  |46.98  |10        |40             |
|fridge      |52.17  |10        |55             |
|microwave   |31.16  |10        |2.25           |

##### Conclusions

Our hypothesis is refuted as results are worse than without different standardization in train and test. To understand better why this happened we have calculated the standard deviation of the fridge for House 1 and House 2 to see if the values are within the same region of consumption. 

![WhatsApp Image 2021-04-17 at 12 28 06](https://user-images.githubusercontent.com/71388892/115109869-ab136900-9f78-11eb-86d9-70b30a624e3e.jpeg)

As it can be seen in this box diagram the consumptions of the fridge for house 1 and house 2 don't follow a similar distribution. Therefore, now it is understandable why the results of Experiment 2 are worse than Experiment 1. We cannot standardize within different values because their consumption don't follow a similar distribution.

#### Experiment 3. Paper with standarization - Using training standardization in test

##### Hypothesis
We wanted to do the opposite of Experiment 2 to see if the dataset with the standardization of the train and test with the train values gave a better outcome.  
The standarization in the Experiment 3 is done as the following: 

<img width="425" alt="Captura de pantalla 2021-04-17 a las 12 49 21" src="https://user-images.githubusercontent.com/71388892/115110751-43abe800-9f7d-11eb-8cc2-911f353a60c3.png">

Better results than experiment 2 are expected although not necessarily better than experiment 1.

##### Experiment setup

See details of the experiments below. Each of the columns describes a specific option of the previously introduced network architectures and pre/post-processing methods:

| Appliance | Regression | Classification | Standardization| Recalculate mean/std in test| 
|-----------|------------|----------------|----------------|-----------------------------|
| dishwasher|       TRUE |           TRUE |          TRUE  |                       FALSE |
| fridge    |       TRUE |           TRUE |          TRUE  |                       FALSE |
| microwave |       TRUE |           TRUE |          TRUE  |                       FALSE |


##### Results

See attached train and test samples per each of the appliances to interpret end evaluate disaggregation:

![image](https://user-images.githubusercontent.com/7881377/114313306-585c2c00-9af6-11eb-9a89-514cb3949e5c.png)
![image](https://user-images.githubusercontent.com/7881377/114313316-627e2a80-9af6-11eb-8af0-a05e28fe9b4a.png)
![image](https://user-images.githubusercontent.com/7881377/114313336-6f028300-9af6-11eb-86fc-292ea240fcd8.png)

<img width="259" alt="Captura de pantalla 2021-04-16 a las 20 25 23" src="https://user-images.githubusercontent.com/71388892/115068039-fc2c4a00-9ef1-11eb-80e8-cca07e338cf9.png">


See obtained error previously introduced in error metrics section and extra training information:

| Appliance  | MAE   | Nº Epochs| Nº Hours Train|
|------------|-------|----------|---------------|
|dishwasher  |31.19  |10        |32             |
|fridge      |39.67  |10        |55             |
|microwave   |23.72  |10        |2.25           |

##### Conclusions

Our hypothesis is supported as the results are better than with different standardization in train and test (as it is done in Experiment 2). In reference with the Experiment 1 we got also worst results, therefore we could conclude that in this situation and with this dataset standarizing the data is not recommended. 

## Paper modification 1 - Only regression enabled

### Experiment 4. Only regression without standarization

#### Hypothesis

The main hypothesis of this experiment is whether attention can detect the consumption pattern and replace what in previous experiments was the classification branch by modulating the output of the regression branch. 
By extracting the classifier branch, the model prediction is expected to detect the peaks (with the help of attention) but may predict values with the biggest difference to the input consumption than with the classification branch.

#### Experiment setup

See details of the experiments below. Each of the columns describe an specific option of the previously introduced network architectures and pre/post processing methods:

| Appliance | Regression | Classification | Standardization| Recalculate mean/std in test| 
|-----------|------------|----------------|----------------|-----------------------------|
| dishwasher|       TRUE |          FALSE |          FALSE |                       FALSE |
| fridge    |       TRUE |          FALSE |          FALSE |                       FALSE |
| microwave |       TRUE |          FALSE |          FALSE |                       FALSE |

#### Results

See attached train vs loss curve to diagnose performance:

![image](https://user-images.githubusercontent.com/7881377/114513412-70d55f00-9c3a-11eb-935d-b5218f09eec3.png)


See attached train and test samples per each of the appliances to interpret end evaluate disaggregation:

![image](https://user-images.githubusercontent.com/7881377/114315323-c7d61980-9afe-11eb-9e45-1e730fc4661c.png)
![image](https://user-images.githubusercontent.com/7881377/114315334-d45a7200-9afe-11eb-8454-eeee8d1d346d.png)
![image](https://user-images.githubusercontent.com/7881377/114315342-dfad9d80-9afe-11eb-9bf0-bc9b3b33f7fb.png)

<img width="259" alt="Captura de pantalla 2021-04-16 a las 20 25 23" src="https://user-images.githubusercontent.com/71388892/115068039-fc2c4a00-9ef1-11eb-80e8-cca07e338cf9.png">

See obtained error previously introduced in error metrics section and extra training information:

| Appliance  | MAE   | Nº Epochs| Nº Hours Train|
|------------|-------|----------|---------------|
|dishwasher  |24.79  |10        |36             |
|fridge      |29.86  |10        |50.9           |
|microwave   |22.56  |10        |1.9            |

#### Conclusions

The results of this experiment are worse than the original paper. Our set of experiments has the lowest mean absolute error (close to Experiment 1 and Experiment 6).
The main hypothesis was that the attention would improve the performance. Hence, the results were better than expected. Attention learns how to focus on peaks of consumption (much better than in Experiment 1) and gives the model the ability to generalize better than what can be seen in Experiment 1. 
Without the classification branch, attention weights train better and the attention values are bigger than in Experiment 1. By not having the classification branch that modulates the regression output, the attention must learn the changes and focus on the significant changes (changes that in Experiment 1 were handled by classification).
Attention in the fridge focus, on state changes (peaks and on-mode) and state duration. But in the microwave case, attention focuses mainly on the switch on. We conclude that after the peak the model expects a long-term change of consumption and in the microwave case it does not occur. That’s the main difference between microwave and fridge. This hypothesis cannot be applied in the dishwasher, because of the peaks of other appliances during the time it is on (that produces noise).
Lastly, the regression is more sensitive to changes and allows to catch the pattern of the input smoothly. But only having the regression model subtracts the model from the specification. 

### Experiment 5. Only regression with standardization (using calculated standardization in test)

#### Hypothesis

The main difference with Experiment 4 is the addition of the standardization in the dataset of the model. In this experiment, we are applying standardization in both training and test sets after splitting the data. We calculate the mean and std variables of the train and test set and apply it respectively.
The hypothesis is that we will have a better outcome than Experiment 4 because both datasets will be standardized in the same way.

#### Experiment setup

See details of the experiments below. Each of the columns describes a specific option of the previously introduced network architectures and pre/post-processing methods:

| Appliance | Regression | Classification | Standardization| Recalculate mean/std in test| 
|-----------|------------|----------------|----------------|-----------------------------|
| dishwasher|       TRUE |          FALSE |           TRUE |                        TRUE |
| fridge    |       TRUE |          FALSE |           TRUE |                        TRUE |
| microwave |       TRUE |          FALSE |           TRUE |                        TRUE |

#### Results

![image](https://user-images.githubusercontent.com/7881377/114315937-9874dc00-9b01-11eb-9d68-e672eb296853.png)
![image](https://user-images.githubusercontent.com/7881377/114315949-a3c80780-9b01-11eb-903e-e302e6412179.png)
![image](https://user-images.githubusercontent.com/7881377/114316065-2bae1180-9b02-11eb-9e51-20be9ce25466.png)

<img width="259" alt="Captura de pantalla 2021-04-16 a las 20 25 23" src="https://user-images.githubusercontent.com/71388892/115068039-fc2c4a00-9ef1-11eb-80e8-cca07e338cf9.png">

See obtained error previously introduced in error metrics section and extra training information:

| Appliance  | MAE   | Nº Epochs| Nº Hours Train|
|------------|-------|----------|---------------|
|dishwasher  |38.78  |10        |36             |
|fridge      |36.38  |10        |51.1           |
|microwave   |23.92  |10        |2.2            |

#### Conclusions

Our hypothesis is not supported as results are worse (significantly in the dishwasher and fridge). 
These results must be produced because the properties (mean and standard deviation) are different in each dataset. So, we are applying different rescaling and making the difference bigger.

### Experiment 6. Only regression with standardization (using training standardization in test)

#### Hypothesis
This experiment combines the only regression architecture with the standardization technique. In this case, we are applying the standardization for the testing set in terms of the mean and standard deviation of the training set. We expect it to improve the results of Experiment 5 given the outcome of Experiments 2 and 3.

#### Experiment setup

See details of the experiments below. Each of the columns describes a specific option of the previously introduced network architectures and pre/post-processing methods:

| Appliance | Regression | Classification | Standardization| Recalculate mean/std in test| 
|-----------|------------|----------------|----------------|-----------------------------|
| dishwasher|       TRUE |          FALSE |           TRUE |                       FALSE |
| fridge    |       TRUE |          FALSE |           TRUE |                       FALSE |
| microwave |       TRUE |          FALSE |           TRUE |                       FALSE |

#### Results

See attached train and test samples per each of the appliances to interpret end evaluate disaggregation:

![image](https://user-images.githubusercontent.com/7881377/114316647-b09a2a80-9b04-11eb-8a76-6c11f0f5f88d.png)
![image](https://user-images.githubusercontent.com/7881377/114316663-bee84680-9b04-11eb-8f75-f66db4c007da.png)
![image](https://user-images.githubusercontent.com/7881377/114450835-c9731080-9bd6-11eb-97aa-9cfd69912d1b.png)

<img width="259" alt="Captura de pantalla 2021-04-16 a las 20 25 23" src="https://user-images.githubusercontent.com/71388892/115068039-fc2c4a00-9ef1-11eb-80e8-cca07e338cf9.png">


See obtained error previously introduced in error metrics section and extra training information:

| Appliance  | MAE   | Nº Epochs| Nº Hours Train|
|------------|-------|----------|---------------|
|dishwasher  |26.37  |10        |36             |
|fridge      |29.96  |10        |51.1           |
|microwave   |20.1   |10        |2.2            |

#### Conclusions
In general, the results are similar to the other experiments in terms of MAE score. In comparison with Experiment 5, the results are significantly better. In Experiments 2 and 3, using training standardization in the test set gave better results as well. 

## Paper modification 2 - Regression and classification using the attention results

### Experiment 7. Using attention in regression and classification

#### Hypothesis
Concatenating the output of the attention layers with the current input of the MLP in the classification branch will affect the prediction of this branch. As we have observed in previously ran experiments, the attention scores peak when the power consumption of the house changes. Consequently, this information can help the classifier decide whether it is a change of consumption or not.

#### Experiment setup

See details of the experiments below. Each of the columns describes a specific option of the previously introduced network architectures and pre/post-processing methods:

| Appliance | Regression | Classification | Standardization| Recalculate mean/std in test| Attention Classification | 
|-----------|------------|----------------|----------------|-----------------------------|--------------------------|
| dishwasher|       TRUE |           TRUE |          FALSE |                       FALSE |                     TRUE |
| fridge    |       TRUE |           TRUE |          FALSE |                       FALSE |                     TRUE |
| microwave |       TRUE |           TRUE |          FALSE |                       FALSE |                     TRUE |

#### Results
See attached train and test samples per each of the appliances to interpret end evaluate disaggregation:
![image](https://user-images.githubusercontent.com/75752252/114435689-bc4d2600-9bc4-11eb-8d4f-c2bdd0ec33c4.png)
![image](https://user-images.githubusercontent.com/75752252/114568196-38eb0d80-9c74-11eb-830d-21afdc2060bb.png)
![image](https://user-images.githubusercontent.com/75752252/114740961-9f425f80-9d4a-11eb-9419-3f17dee8e397.png)

<img width="259" alt="Captura de pantalla 2021-04-16 a las 20 25 23" src="https://user-images.githubusercontent.com/71388892/115068039-fc2c4a00-9ef1-11eb-80e8-cca07e338cf9.png">

See obtained error previously introduced in error metrics section and extra training information:

| Appliance  | MAE   | Nº Epochs| Nº Hours Train|
|------------|-------|----------|---------------|
|dishwasher  |28.09  |3         |6              |
|fridge      |31.08  |4         |8              |
|microwave   |26.98  |10        |1              |

#### Conclusions
In this case, the results are similar to the other experiments, for upcoming experimentation, what we would propose to use the attention output to calculate the classification would be eliminating this branch's convolutional layers. The magnitude of the values of the concatenated vector that enters the MLP can differ between the ones coming from attention and the ones coming from CNN layers, so this can handicap the training process.

One observation that can be made is that in the case of the dishwasher and the fridge, overfitting started in the 3rd and 4th epochs, as can be noted in the image below (dishwasher).

![results-error](https://user-images.githubusercontent.com/75752252/114723899-546d1b80-9d3b-11eb-969e-a7f3ba37015d.png)


### Experiment results summary

![image](https://user-images.githubusercontent.com/75752252/114727977-f4787400-9d3e-11eb-8355-df2109c81dcc.png)

## Implementation details

- Files description
  - settings.yaml. YAML file describing each of the experiment parameters
    - Train, val and test dataset properties
    - Hyperparameters (oversampling factor, learning rate, window size, filter properties, ...)
  - redd.yaml YAML file describing REDD dataset files and parameters (building and channels filenames)
  - redd.py REDD dataset parser
  - dataset.py REDD dataset loader and preprocessing
  - run-train.sh Experiments training launcher using default arguments
  - run-test.sh Experiments testing launcher using default arguments
  - main.py Orchestrator of train and test actions. Multiple arguments supported to handle different experiments actions and scenarios
  - model.py Described models implementation
  - train.py Train handler. Manage multiple epoch training and evaluation on training dataset
  - test.py Test handler. Manage pre-trained model and testing on testing dataset
  - utils.py Data handlers, error and plot helping functions 
- Framework
  - Python3.7.9
  - Torch 1.7.1
- Computing resources
  - Using pre-configured Cloud Deep Learning VM from Google Cloud Market
   - vCPU cores: 2
   - RAM: 13 GB
   - 1 NVIDIA Tesla K80

In order to run the code follow instructions below:

```
1) Clone the github project
   git clone https://github.com/abadiabosch/dlai-upc-2021-project.git

2) Install python requirements
   pip install -r requirements.txt

3) Download the dataset
   https://drive.google.com/drive/folders/1ey1UBfU41zjftiXjp6PmJ0OfXFhJYj4N?usp=sharing
   
4) Update settings.yaml to make dataset.path field point to the folder with *.csv downloaded in step 3
   dataset:
       path: <path-to-dataset>

5) To train models run command below in source folder. Using default training settings, see main orchestrator arguments below
   sh run-train.sh
   
6) To test models rename output-train folder to output-test folder and run command below in source folder. Using default testing settings,
   see main orchestraro arguments below
   sh run-test.sh
   
Main orchestrator command line arguments. See default settings in train and test launchers

    Command line arguments parser
    --settings
      Path to settings yaml file where all disaggregation scenarios and model hyperparameters are described
    --appliance
      Name of the appliance to train or test
    --path
      Path to output folder where resuls are saved
    --train
      Set to train or unset to test
    --tune
      Set to enable automatic architecture hyperparameters tunning 
    --epochs
      Number of epochs to train
    --disable-plot
      Disable sliding window plotting during train or test
    --disable-random
      Disable randomness in processing

```

In order to run pre-trained models follow additional instructions below:
```
1) Download models from
   https://drive.google.com/drive/folders/1gb_FmG1hs6lgSlSF9MLZ4w7rAgNEtfvC?usp=sharing

2) Copy each of the models (<appliance>.th) to its path described in run-train.sh or run-test.sh
```

## Conclusions

-	Conclusion 1: During the experiments; the results were significantly better in train and eval than in test. The explanation behind this outcome is that our model was trained with a dataset  with a very low variation of patterns of appliances. This is because there were just 3 to 5 different types of the same appliance for training, to test the model for a totally different type of appliance. For example, the patterns of the fridge consumption in the training set were different from the pattern of the testing set, therefore, the model did not have a broad variety of load profile patterns to learn to infer from. In the next image, we can see the variation of one house from the training set and the house for the testing set for the fridge appliance. The variation is of the two houses is totally different, with this graphic our explanation is endorsed.  

![WhatsApp Image 2021-04-17 at 12 45 21](https://user-images.githubusercontent.com/71388892/115111764-288fa700-9f82-11eb-845d-6bcef89c28e5.jpeg)

-	Conclusion 2: Classification is in charge of modeling the real consumption patterns of each window given.
-	Conclusion 3: Regression is in charge of infering the maximum consumption of the appliance in the input window. This applies to all the models but the "only regresion" one, as explained in conclusion 6.
-	Conclusion 4: Attention focuses on two scenarios related to the simultaneity of the appliances: 
	 - Scenarios with a high simultaneity factor: Attention focus on State changes of appliances (switch on/off) and state duration. Therefore, focuses on the appliance pattern. (case of the microwave and dishwasher)
	 - Scenarios with low simultaneity factor: attention focuses on the neighborhood, outside of the active section of the appliance. (case of the fridge)
-	Conclusion 5: We don’t have clear conclusions of whether the standardization of the data set will produce better outcomes than the paper reference model.
-	Conclusion 6: Without a classification branch, the output is more smooth and therefore it does not capture adequately the peaks of consumption. This is because the regression branch is not prepared to do both tasks of inferring the maximum consumption and adapting to the exact pattern with instantaneous changes of power.
-	Conclusion 7: All the models took a big amount of time to be trained, the amount of data, the complexity of the forward and backward processes and the computational resources were the reason for that.

## Future work

Transformers are state-of-the-art models with a high impact in deep learning. To continue developing the project we wanted to add this new attention mechanism to our model. Unfortunately, we did not have time to develop it. 
Therefore, as future work, we recommend applying transformers in replace to the attention layer in our model.
The encoder module will use the input from the LSTM and will feed the self-attention block to reach the 1D convolution. A residual connection and layer normalization would be implemented.
The encoder would follow a similar procedure as the encoder, adding a cross-attention. The cross attention would find which regions in the input consumption sequence are most relevant to constructing and therefore deserve the highest attention coefficients.
Our hypothesis after applying transformers is to generate a better outcome than with the actual model. Being the model more efficient when selecting the regions in which the consumption sequence varies. 

## References

<a id="1">[1]</a> 
Rolnick, D., Donti, P. L., Kaack, L. H., Kochanski, K., Lacoste, A., Sankaran, K., ... & Bengio, Y. (2019).
Tackling climate change with machine learning.
arXiv preprint arXiv:1906.05433.
[https://arxiv.org/abs/1906.05433](https://arxiv.org/abs/1906.05433)

<a id="2">[2]</a> 
Piccialli, V., & Sudoso, A. M. (2021)
Improving Non-Intrusive Load Disaggregation through an Attention-Based Deep Neural Network.
Energies, 14(4), 847.
[https://arxiv.org/abs/1912.00759](https://arxiv.org/abs/1912.00759)

<a id="3">[3]</a> 
Zhao, B., Ye, M., Stankovic, L., & Stankovic, V. (2020).
Non-intrusive load disaggregation solutions for very low-rate smart meter data.
Applied Energy, 268, 114949.
[https://www.sciencedirect.com/science/article/abs/pii/S030626192030461X](https://www.sciencedirect.com/science/article/abs/pii/S030626192030461X)

<a id="4">[4]</a> 
Zhang, C., Zhong, M., Wang, Z., Goddard, N., & Sutton, C. (2018, April).
Sequence-to-point learning with neural networks for non-intrusive load monitoring.
In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 32, No. 1).
[https://arxiv.org/abs/1612.09106]](https://arxiv.org/abs/1612.09106)

<a id="5">[5]</a> 
Kolter, J. Z., & Johnson, M. J. (2011, August)
REDD: A public data set for energy disaggregation research.
In Workshop on data mining applications in sustainability (SIGKDD), San Diego, CA (Vol. 25, No. Citeseer, pp. 59-62).
[http://redd.csail.mit.edu/kolter-kddsust11.pdf](http://redd.csail.mit.edu/kolter-kddsust11.pdf)

<a id="6">[6]</a> 
Shin, C., Joo, S., Yim, J., Lee, H., Moon, T., & Rhee, W. (2019, July).
Subtask gated networks for non-intrusive load monitoring.
In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 33, No. 01, pp. 1150-1157).
[https://www.mdpi.com/1996-1073/14/4/847/pdf](https://www.mdpi.com/1996-1073/14/4/847/pdf)
