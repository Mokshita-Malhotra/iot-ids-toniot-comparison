# Comparing machine learning classifiers for IoT intrusion detection (ToN-IoT)

Code for the paper "A Comparison of Machine Learning Classifiers for Intrusion Detection on the ToN-IoT Network Dataset" (Mokshita Malhotra, Suryansh Gupta).

The project compares logistic regression, k-nearest neighbours, decision tree, random forest, gradient boosting and XGBoost on the ToN-IoT network traffic, for binary detection (normal or attack) and ten-class attack identification. It also includes SHAP explanations, a destination-port ablation, bootstrap confidence intervals, McNemar tests, a class-weighting experiment for the rare mitm class, and a small neural network baseline.

## Dataset

The data is not included in this repository. Download `Train_Test_Network.csv` from the official ToN-IoT page at UNSW Canberra (https://research.unsw.edu.au/projects/toniot-datasets) and place it in the same folder as the scripts. Please cite the dataset authors if you use it:

- N. Moustafa, "A new distributed architecture for evaluating AI-based security systems at the edge: Network TON_IoT datasets," Sustainable Cities and Society, vol. 72, 2021.
- T. M. Booij et al., "ToN_IoT: The role of heterogeneity and the need for standardization of features and attack types in IoT network intrusion data sets," IEEE Internet of Things Journal, vol. 9, no. 1, 2022.

## How to run

The scripts were run on a Google Colab CPU runtime. Install the packages first:

```
pip install -r requirements.txt
```

Run the scripts in this order. Keep `SAMPLE = 150000` and `SEED = 42` the same in all of them so every script uses the same test set.

1. `toniot_experiment.py` : main comparison of the six models (results_binary.csv, results_multiclass.csv, confusion matrices, feature importance).
2. `explainability.py` : SHAP and permutation importance for XGBoost.
3. `extra_experiments.py` : ablation, confidence intervals, McNemar tests, per-class metrics, rare-class experiment, MLP baseline.

The full runs take from about 15 minutes to a couple of hours on a free Colab CPU, mostly because of gradient boosting on the ten-class task. Lower `SAMPLE` in the script settings for a quick test.

## Repository layout

- `code/` : the Python scripts
- `results/` : CSV output files
- `figures/` : confusion matrices, feature importance and SHAP plots

## Notes

- IP addresses, source port and timestamp are dropped on purpose, because the attacking machines have fixed addresses in this dataset and keeping them lets models score well without learning attack behaviour.
- Results can differ slightly between library versions and hardware. Training times depend on the shared Colab machine.

## License

MIT License (see `LICENSE`). The dataset has its own terms, see the UNSW page above.
