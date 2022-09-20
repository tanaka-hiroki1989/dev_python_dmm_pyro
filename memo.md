# dmm_pyro
----
## 環境構築
```
conda env create --file dmm_pyro_env.yml
```

 
## データセット
The JSB chorales are a set of short, four-voice pieces of music well-noted for their stylistic homogeneity. 
The chorales were originally composed by Johann Sebastian Bach in the 18th century. 
He wrote them by first taking pre-existing melodies from contemporary Lutheran hymns and then harmonising them to create the parts for the remaining three voices. 
The version of the dataset used canonically in representation learning contexts consists of 382 such chorales, with a train/validation/test split of 229, 76 and 77 samples respectively.
- https://paperswithcode.com/dataset/jsb-chorales

https://github.com/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/tree/master/deep-learning/UBER-pyro/examples/dmm

## loss and guide
    ```
    if args.tmc:
        if args.jit:
            raise NotImplementedError("no JIT support yet for TMC")
        tmc_loss = TraceTMC_ELBO()
        dmm_guide = config_enumerate(
            dmm.guide,
            default="parallel",
            num_samples=args.tmc_num_samples,
            expand=False,
        )
        svi = SVI(dmm.model, dmm_guide, adam, loss=tmc_loss)
    elif args.tmcelbo:
        if args.jit:
            raise NotImplementedError("no JIT support yet for TMC ELBO")
        elbo = TraceEnum_ELBO()
        dmm_guide = config_enumerate(
            dmm.guide,
            default="parallel",
            num_samples=args.tmc_num_samples,
            expand=False,
        )
        svi = SVI(dmm.model, dmm_guide, adam, loss=elbo)
    else:
        elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
        svi = SVI(dmm.model, dmm.guide, adam, loss=elbo)
    ```


Finally, we should mention that the main difference between the DMM implementation described here and the one used in reference [1] is that they take advantage of the analytic formula for 
*the KL divergence between two gaussian distributions* (whereas we rely on Monte Carlo estimates). 

This leads to lower variance gradient estimates of the ELBO, which makes training a bit easier. 

We can still train the model without making this analytic substitution, but training probably takes somewhat longer because of the higher variance. To use analytic KL divergences use *TraceMeanField_ELBO*

- tmcELBO

A trace-based implementation of Tensor Monte Carlo [1]
by way of Tensor Variable Elimination [2] that supports:
  - local parallel sampling over any sample site in the model or guide
  - exhaustive enumeration over any sample site in the model or guide
To take multiple samples, mark the site with
    ``infer={'enumerate': 'parallel', 'num_samples': N}``.
To configure all sites in a model or guide at once,
use :func:`~pyro.infer.enum.config_enumerate` .
To enumerate or sample a sample site in the ``model``,
mark the site and ensure the site does not appear in the ``guide``.

This assumes restricted dependency structure on the model and guide:
variables outside of an :class:`~pyro.plate` can never depend on
variables inside that :class:`~pyro.plate` .

    References

    [1] `Tensor Monte Carlo: Particle Methods for the GPU Era`,
        Laurence Aitchison (2018)

    [2] `Tensor Variable Elimination for Plated Factor Graphs`,
        Fritz Obermeyer, Eli Bingham, Martin Jankowiak, Justin Chiu, Neeraj Pradhan,
        Alexander Rush, Noah Goodman (2019)
    """