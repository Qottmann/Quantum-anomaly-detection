# Quantum Anomaly Detection: Unsupervised mapping of phase diagrams on a physical quantum computer

Entry for the [Qiskit Europe Hackathon](https://qiskithackathoneurope.bemyapp.com/#/projects/60805383ff00f400197f84fd)

We propose quantum anomaly detection (QAD), a novel quantum machine learning framework for exploring phase diagrams of quantum many-body systems. QAD is trained in a fully unsupervised fashion on a quantum device. The implentation is based on [Qiskit](https://qiskit.org/). We walk you through our proposal in [main.ipynb](main.ipynb).

* [qae.py](qae.py): Includes `QAEAnsatz`, the parameterized circuit ansatz for the QAD, and `QAE`, the training framework
* [experiments/](experiments/): The experiments performed on the physical device. Each file has a short description of the experiment in the beginning
* [main.ipynb](main.ipynb): A complete run through our proposal showcasing all its ingredients. In its form here it is generating the result for the 2D Antiferromagnetig Ising model in real-noise simulations
* [simulations/qae-from-vqe.ipynb](simulations/qae-from-vqe.ipynb): The QAE results for the BH model
* [data/](data/): Our notebooks are configured in the way that all results are stored in numpy files for reuse
* [plotting/](plotting/): Notebooks to replot the results in the desired fashion for the presentation and paper
