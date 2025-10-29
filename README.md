# TGGNN4ACOPF
This repo contains source code accompanying the '[Towards Generalization of Graph Neural Networks for AC Optimal Power Flow](https://arxiv.org/abs/2510.06860)' paper. The code is organized according to the case studies in the paper. The data to reproduce the results in the paper can be found on [Zenodo](https://zenodo.org/uploads/17476069). The 2000-bus grid datasets are unfortunately not included due to data size limits on Zenodo. The scripts in the `data converter` folder can be used to convert the individual JSON files from [OPFData](https://arxiv.org/abs/2406.07234) into a combined Numpy compressed file. 
Experiment results are logged in wandb. Please provide your own wandb api key and chnage the paths to data input to run the notebooks. 

If you use the data or code in your work, please cite:

```
@misc{arowolo2025generalizationgraphneuralnetworks,
      title={Towards Generalization of Graph Neural Networks for AC Optimal Power Flow}, 
      author={Olayiwola Arowolo and Jochen L. Cremer},
      year={2025},
      eprint={2510.06860},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.06860}, 
}
```
