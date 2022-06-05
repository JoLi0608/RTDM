[//]: # (Image References)


# Real-time Decision Making

This repostory contains the code for figures in thesis 'Real-time Decision Making'.

## Description

### data

The folder $data$ contains two .pkl files storing the data for our real-time benchkark.

inference_time.pkl stores the median inference delay of selected algorithms in different continuous-control tasks under varing hardware conditions.

repeat.pkl stores the reward achieved by selected algorithms with varying delay/sticky actions.

### scripts

evaluation.py is the script to evaluate trained algorithms under real-time settings.

inference_plot.py uses the inference.pkl to plot the result of inference delay analysis, correspondind to Figures 4.1  4.2 B.2 and B.1 in the thesis.

performance_plot.py uses both inference.pkl and repeat.pkl to plot the performance degradation against sticky actions/delay and varing computation resources, Corresponding to Fogures 4.3 4.4 4.5 and B.3 in the thesis.

### references

the resources of trained algorithms and selected environments are the same as specified in the thesis.


