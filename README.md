# diffseg
An implementation of the relaxed segmented model with TSP-based warping functions. The methodology is described and analysed in detail in the paper:

> Erik Scharwächter, Jonathan Lennartz and Emmanuel Müller: **Differentiable Segmentation of Sequences.** In: Proceedings of the International Conference on Learning Representations (ICLR), 2021. [[OpenReview]](https://openreview.net/forum?id=4T489T4yav) [[arXiv]](https://arxiv.org/abs/2006.13105)

We provide a simple Python module [nwarp.py](./nwarp.py) that contains PyTorch modules for all required components. The Jupyter notebooks demonstrate how to use the components for a large variety of tasks: Poisson regression on COVID-19 data ([eval-covid19.ipynb](./eval-covid19.ipynb)), change point detection ([eval-gaussiancp.ipynb](./eval-gaussiancp.ipynb)), classification under concept drift ([eval-conceptdrift.ipynb](./eval-conceptdrift.ipynb)), and phoneme segmentation  ([eval-timit.ipynb](./eval-timit.ipynb)).

## Contact and Citation

* Corresponding author: [Erik Scharwächter](mailto:erik.scharwaechter@cs.tu-dortmund.de)
* Please cite our paper if you use or modify our code for your own work. Here's a `bibtex` snippet:

```
@inproceedings{Scharwachter2021,
    author = {Scharw{\"{a}}chter, Erik and Lennartz, Jonathan and M{\"{u}}ller, Emmanuel},
    booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
    title = {{Differentiable Segmentation of Sequences}},
    year = {2021}
}
```
## Requirements

The `nwarp` module itself requires `torch` and [`libcpab`](https://github.com/SkafteNicki/libcpab) (for the CPA-based warping functions). The Jupyter notebooks have additional dependencies that can be checked in the respective files.

## License

The source codes are released under the [MIT license](./LICENSE). The data in [RKI_COVID19.csv](./RKI_COVID19.csv) are published with the title "Fallzahlen in Deutschland" by [Robert Koch Institute (RKI)](https://www.rki.de/) under the [Data licence Germany – attribution – Version 2.0 (dl-de/by-2-0)](./RKI_COVID19.LICENSE)
