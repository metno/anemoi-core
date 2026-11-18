.. _spread-calibration:

########################################
 Spread calibration of ensemble models
########################################

An ensemble trained with a proper score such as the almost-fair CRPS
calibrates its spread to the error it meets on the training samples.
Where the model has partly memorised its training set, that error is
smaller than the error on unseen data, and the ensemble is
under-dispersed out of sample although it looks reliable in training.
The signature is a spread–skill ratio that is close to 1 on training
batches and clearly below 1 on validation data.

The tools in :mod:`anemoi.training.spread` measure this gap during
training and remove it in a short fine-tune of an existing model.

************
 Components
************

``SpreadSkillLogger`` and ``TrainingDiagnosticsLossWrapper``
   Train-side and validation-side ensemble variance, squared error of
   the ensemble mean and the corrected spread–skill ratio
   ``sqrt((M+1)/M · <s²> / <(x̄ − y)²>)``, per variable, rollout step
   and (with the multiscale spread loss) scale band. Logging only: the
   wrapper returns exactly the wrapped loss.

``TargetNoiseEnsembleTraining``
   Training method that evaluates the loss against targets perturbed
   with spatially coherent Gaussian noise, resampled at every visit.
   With the amplitude ``A² = c · MSE_val − MSE_train`` per variable and
   rollout step, the training-side error is raised to the level the
   model meets out of sample, and the score itself stays proper. This
   is the component that closes the gap.

``MultiscaleSpreadLossWrapper`` and ``SpreadScaledCRPS``
   The multiscale loss with a spread factor ``a`` per scale band,
   variable and rollout step: each band is scored after widening the
   ensemble about its mean by ``a``. With all factors at 1 it is the
   stock multiscale loss.

``CombinedAdaptiveSpreadController``
   Feedback on those factors towards a validation spread–skill ratio of
   1, within clamps. Next to target noise it acts as a small two-sided
   correction. On its own the widening did not close the gap in our
   experiments, and a large static factor costs skill, since the score
   is proper only at ``a = 1``.

********
 Recipe
********

#. Fine-tune the model at the intended rollout depth with its plain
   loss and the logger active (a control run). Its epoch dump holds
   the training and validation errors per variable and rollout step.

#. Build the amplitude tables from that dump::

      python tools/spread_calibration/make_noise_tables.py --dump <dump.jsonl> \
          --steps 1 2 3 4 --out-prefix tables/target_noise_step

#. Build the smoothing matrices of the loss bands on the model grid::

      python tools/spread_calibration/build_smoothers.py --graph <graph.pt> --out smoothers \
          --smoother gauss_8x:8:2.0 --smoother gauss_2x:2:2.5

#. Optionally provide a dose map, an ``.npz`` with ``names`` (variables),
   ``w`` (variable × grid point, area mean of ``w²`` equal to 1) and
   ``lat``, to move noise between regions whose gaps differ.

#. Fine-tune the original model again with target noise, the multiscale
   spread loss and the controller:

.. literalinclude:: yaml/example_spread_calibration.yaml
   :language: yaml

*******
 Notes
*******

-  The source of the tables must have trained against clean targets: a
   target-noise run's own training error includes the noise.

-  Train-side diagnostics of a target-noise run are computed against
   the noised targets. The validation-side ratio is the reference.

-  Per-band diagnostics, and therefore the controller, are not
   available when one model is sharded over several GPUs.

-  The recipe was developed and verified on an O96 model as a rollout-4
   fine-tune of 10 000 to 20 000 steps. Chained smoothing matrices for
   finer grids are included as an experimental feature.
