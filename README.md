# QML_Network_Anomaly_Detection
QML model implementation for network packet anomaly detection. In this project I've tried to test the performances of different type of QML models, using libraries like *qiskit* and *qiskit_machine_learning*, to see how a Quantum Model can be efficient with real datasets. The entire work comprises a total of 3 scenarios:
1. Trivial scenario: a trivial QML model is implemented to perform anomaly detection over a trivial dataset of only 4 features, using QVC algorithm.
2. Scaled scenario: a QSVM model is used over a scaled up dataset, this time with 10 features.
3. Real scenario: a QVC model is used to perform anomaly detection over a real dataset of data packets, to find realistic attack patterns.
The last scenario is implemented using the **UNSW_NB15_testing-set.csv** and the **UNSW_NB15_training-set.csv** files, which are datasets implemented by the Australian Centre for Cybersecurity, and they consists of a total of 49 features.
The final goal is to find analyze network packets, either fake or real ones, to find some attack pattern for a trivial Intrusion Detection System.
## Structure of the project
- **dataset.py**: file with functions used to generate different type of sample dataset for our experiments.
- **my_qml_model**: jupyter notebook with several experiments with different QML models in different scenarios.
- **UNSW_NB15_testing-set.csv**: real testing data
- **UNSW_NB15_training-set.csv**: real training data
