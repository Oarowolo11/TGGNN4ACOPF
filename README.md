# TGGNN4ACOPF
This repo contains source code accompanying the '[Towards Generalization of Graph Neural Networks for AC Optimal Power Flow](https://arxiv.org/abs/2510.06860)' paper. The code is organised according to the case studies in the paper. The data to reproduce the results in the paper can be found on [Zenodo](https://zenodo.org/uploads/17476069). The 2000-bus grid datasets are unfortunately not included due to data size limits on Zenodo. The scripts in the `data converter` folder can be used to convert the individual JSON files from [OPFData](https://arxiv.org/abs/2406.07234) into a combined NumPy compressed file. 
Experiment results are logged in wandb. Please provide your own wandb api key and change the paths to the data input to run the notebooks. 

If you use the data or code in your work, please cite:

```
@ARTICLE{Arowolo2026-dz,
  title     = "Towards generalization of graph neural networks for {AC} optimal
               power flow",
  author    = "Arowolo, Olayiwola and Cremer, Jochen L",
  journal   = "Energy and AI",
  publisher = "Elsevier BV",
  volume    =  25,
  number    =  100842,
  pages     = "100842",
  month     =  sep,
  year      =  2026,
  copyright = "http://creativecommons.org/licenses/by/4.0/",
  language  = "en"
}
```
