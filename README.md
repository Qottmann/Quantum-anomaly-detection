# Quantum Anomaly Detection: Unsupervised mapping of phase diagrams on a physical quantum computer

Entry for the [Qiskit Europe Hackathon](https://qiskithackathoneurope.bemyapp.com/#/projects/60805383ff00f400197f84fd)

We propose quantum anomaly detection (QAD), a novel quantum machine learning framework for exploring phase diagrams of quantum many-body systems. QAD is trained in a fully unsupervised fashion on a quantum device. The implentation is based on [Qiskit](https://qiskit.org/).

* [qae.py](qae.py): Includes `QAEAnsatz`, the parameterized circuit ansatz for the QAD, and `QAE`, the training framework
* [ibmq_antiferro-1D_bogota_results.ipynb](ibmq_antiferro-1D_bogota_results.ipynb): The results of the actual quantum computation
* [clean_VQE_Ising.ipynb](clean_VQE_Ising.ipynb): The VQE simulations for the Ising model
* [qae-from-vqe.ipynb](qae-from-vqe.ipynb): The QAE results for the BH model
