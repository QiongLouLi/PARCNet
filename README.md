# PARCNet

This repository is the official implementation of the paper **PARCNet: Phase-Aware Residual Correction Network for Efficient Multivariate Time Series Forecasting**.

📄 This paper has been accepted by **Knowledge-Based Systems (KBS)**.
🔗 Paper: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0950705126010968)

## Method Overview

**Motivation**: In long-horizon multivariate time series forecasting, the series usually contain multiple periodic components, and predictions are prone to phase misalignment with the ground truth, so errors accumulate as the forecasting horizon grows.

**Key idea**: PARCNet corrects forecasts from the perspective of **phase**. It extracts amplitude and phase features based on the **Hilbert analytic signal**, and then refines the predictions of a backbone model through a **1D convolutional gating** mechanism and a **phase-aware residual correction** module, alleviating periodic misalignment.

**Highlights**:
- **Phase-aware**: explicitly models amplitude and phase information via the Hilbert analytic signal;
- **Plug-and-play residual correction**: can be attached to different backbone forecasting models without changing their original structure;
- **Efficient**: the correction module is lightweight and adds little overhead;
- **Strong performance**: consistently improves the forecasting accuracy of backbone models on multiple public benchmarks.

![PARCNet framework](![Uploading image.png…]()
)

## Citation

If you find this repository useful for your research, please consider citing our paper:

```bibtex
@article{parcnet,
  title   = {PARCNet: Phase-Aware Residual Correction Network for Efficient Multivariate Time Series Forecasting},
  author  = {Li, Ziqiong and ...},
  journal = {Knowledge-Based Systems},
  year    = {2026}
}
```

## Contact

If you have any questions, please feel free to open an issue or contact the authors:

* Ziqiong Li
  Central South University
  Email: [qionglouyuyu@csu.edu.cn](mailto:qionglouyuyu@csu.edu.cn)
