<div align="center">

# Creating and running evaluations

</div>

The evaluation pipeline consists of an evaluator (specific to a benchmark) which takes a model and a group of evaluation metrics, computes and reports the evaluations. The evaluation settings are stored in experiment configs which can be used off-the-shelf.

We discuss full details of creating metrics in [#metrics](#metrics) and benchmarks in [#benchmarks](#benchmarks).


## Quick evaluation
Run the TOFU benchmark evaluation on a checkpoint of a LLaMA 3.2 model:
```bash
python src/eval.py --config-name=eval.yaml \
  experiment=eval/tofu/llama2 \ 
  model=Llama-3.2-3B-Instruct \ 
  model.model_args.pretrained_model_name_or_path=<LOCAL_MODEL_PATH> \
  task_name=SAMPLE_EVAL
```
- `--config-name=eval.yaml`- sets task to be [`configs/eval.yaml`](../configs/eval.yaml)
- `experiment=eval/tofu/default`- set experiment to use [`configs/eval/tofu/default.yaml`](../configs/eval/tofu/default.yaml)
- `model=Llama-3.2-3B-Instruct`- override the default (`Llama-3.2-1B-Instruct`) model config to use [`configs/model/Llama-3.2-3B-Instruct`](../configs/model/Phi-3.5-mini-instruct.yaml).
- Output directory: constructed as `saves/eval/SAMPLE_EVAL`


Run the MUSE-Books benchmark evaluation on a checkpoint of a Phi-3.5 model:
```bash
python src/eval.py --config-name=eval.yaml \
  experiment=eval/muse/default \
  data_split=Books
  model=Llama-2-7b-hf.yaml \
  model.model_args.pretrained_model_name_or_path=<LOCAL_MODEL_PATH> \
  task_name=SAMPLE_EVAL
```
- `---config-name=eval.yaml`- this is set by default so can be omitted
- `data_split=Books`- overrides the default MUSE data split (News). See [`configs/experiment/eval/muse/default.yaml`](../configs/experiment/eval/muse/default.yaml)

## Metrics

A metric takes a model and a dataset and computes statistics of the model over the datapoints (or) takes other metrics and computes an aggregated score over the dataset.

### Forget probability (formal definition)

**Forget probability** is the probability metric evaluated on the **forget set** $\mathcal{D}_f$ (e.g. TOFU’s forget Q&A split). It measures how likely the model is to assign high probability to the *answer* tokens of forget-set examples; lower values indicate better unlearning.

**Per-example score.** For a single example $i$ with input_ids $\mathbf{x}_i$ and labels $\mathbf{y}_i$, only positions where the label is not ignored (e.g. answer tokens) are used. Let $T_i$ be the set of sequence positions $t$ with non-ignored labels. The model’s next-token distribution at each position is $p_\theta(\cdot \mid \mathbf{x}_i, \mathbf{y}_{i,<t})$. The per-example average cross-entropy over the target tokens is

$$
\bar{L}_i = \frac{1}{|T_i|} \sum_{t \in T_i} \bigl( -\log p_\theta(y_{i,t} \mid \mathbf{x}_i, \mathbf{y}_{i,<t}) \bigr).
$$

The **per-example (normalized) probability** is

$$
P_i = \exp(-\bar{L}_i) = \left( \prod_{t \in T_i} p_\theta(y_{i,t} \mid \mathbf{x}_i, \mathbf{y}_{i,<t}) \right)^{1/|T_i|},
$$

i.e. the **geometric mean** of the model’s next-token probabilities over the labeled (answer) positions.

**Aggregate (forget probability).** The **forget probability** reported by the evaluator is the mean of these per-example scores over the forget set:

$$
\text{ForgetProb}(\theta; \mathcal{D}_f) = \frac{1}{|\mathcal{D}_f|} \sum_{i \in \mathcal{D}_f} P_i.
$$

**Implementation.** The handler is the `probability` metric ([`src/evals/metrics/memorization.py`](../src/evals/metrics/memorization.py)); the core per-batch computation is `evaluate_probability` in [`src/evals/metrics/utils.py`](../src/evals/metrics/utils.py). When the dataset is the forget split (e.g. via [`configs/eval/tofu_metrics/forget_Q_A_Prob.yaml`](../configs/eval/tofu_metrics/forget_Q_A_Prob.yaml)), the reported `agg_value` is the forget probability above; `value_by_index` gives the per-example $P_i$.

### Confidence-ordered forget probability (diffusion LLMs)

For **diffusion language models (dLLMs)**, tokens are not generated in causal order; the model predicts (and “fixes”) positions bidirectionally, often **setting the top-confidence labels first**. The **confidence-ordered forget probability** is the forget-probability analogue defined by ordering positions by the model’s **confidence in the true label** at each position, then aggregating in that order.

**Confidence (per position).** For a single example $i$ with input $\mathbf{x}_i$ and labels $\mathbf{y}_i$, let $T_i$ be the set of labeled (non-ignored) positions. At each position $t \in T_i$, the model (in one forward pass) produces a distribution over the vocabulary. The **confidence** for the true label $y_{i,t}$ at position $t$ is

$$
c_{i,t} = p_\theta(y_{i,t} \mid \mathbf{x}_i, \mathbf{y}_i),
$$

i.e. the probability the model assigns to the correct token at $t$. For a bidirectional/diffusion model this uses full-sequence context (no causal masking).

**Confidence ordering.** Sort the labeled positions by confidence (descending): $t_{(1)}, t_{(2)}, \ldots, t_{(n)}$ so that

$$
c_{i,t_{(1)}} \geq c_{i,t_{(2)}} \geq \cdots \geq c_{i,t_{(n)}}.
$$

We interpret this as “setting the top confidence labels first.”

**Per-example score (confidence-ordered).** The **confidence-ordered forget probability** for example $i$ is the geometric mean of the model’s probabilities for the true labels, **taken in confidence order** (highest-confidence position first):

$$
P_i^{\mathrm{co}} = \left( \prod_{j=1}^{n} p_\theta(y_{i,t_{(j)}} \mid \mathbf{x}_i, \mathbf{y}_i) \right)^{1/n}.
$$

Since the model’s distribution at each position in one forward pass does not depend on the order we write the product, this equals $\bigl( \prod_{t \in T_i} p_\theta(y_{i,t} \mid \mathbf{x}_i, \mathbf{y}_i) \bigr)^{1/n}$ — the same scalar as the (causal) forget probability when using the same per-position probabilities. The difference is **conceptual**: for AR we condition causally ($\mathbf{y}_{<t}$); for dLLMs we use the model’s (bidirectional) probability at each $t$ and **order** the product by confidence to mirror “set top confidence labels first.”

**Aggregate.** The **confidence-ordered forget probability** over the forget set $\mathcal{D}_f$ is

$$
\text{ForgetProb}^{\mathrm{co}}(\theta; \mathcal{D}_f) = \frac{1}{|\mathcal{D}_f|} \sum_{i \in \mathcal{D}_f} P_i^{\mathrm{co}}.
$$

**Optional output.** The implementation can also return the **confidence-ordered sequence** $(p_{t_{(1)}}, p_{t_{(2)}}, \ldots, p_{t_{(n)}})$ per example (in `value_by_index` under `confidence_ordered_probs`) for analysis, e.g. plotting probability vs. confidence rank.

**Implementation.** Handler: `probability_confidence_ordered` ([`src/evals/metrics/memorization.py`](../src/evals/metrics/memorization.py)); core: `evaluate_probability_confidence_ordered` in [`src/evals/metrics/utils.py`](../src/evals/metrics/utils.py). Use the same dataset config as forget probability (e.g. forget split) and select the `probability_confidence_ordered` handler. See [Diffusion LLM Support](diffusion_support.md) for using this metric with diffusion models.

Some metrics are reported as both individual points and aggregated values (averaged): probability scores, ROUGE scores, MIA attack statistics, Truth Ratio scores etc. They return a dictionary which is structured as `{"agg_value": ..., "values_by_index": {"0":..., "1":..., ...}}`.

Other metrics like TOFU's Forget Quality (which is a single score computed over forget v/s retain distributions of Truth Ratio) and MUSE's PrivLeak (which is a single score computed over forget v/s holdout distributions of MIA attack values) aggregate the former metrics into a single score. They return a dictionary which contains `{"agg_value": ...}`.

### Steps to create new metrics:

#### 1. Implement a handler
Metric handlers are implemented in [`src/evals/metrics`](../src/evals/metrics/), where we define handlers for `probability`, `rouge`, `privleak` etc.

A metric handler is implemented as a function decorated with `@unlearning_metric`. This decorator wraps the function into an UnlearningMetric object. This provides functionality to automatically load and prepare datasets and collators for `probability` as specified in the eval config ([example](../configs/eval/tofu_metrics/forget_Q_A_Prob.yaml)), so they are readily available for use in the function.


Example: implementing the `rouge` and `privleak` handlers

```python
# in src/evals/metrics/memorization.py
@unlearning_metric(name="rouge")
def rouge(model, **kwargs):
    """Calculate ROUGE metrics and return the aggregated value along with per-index scores."""
    # kwargs is populated on the basic of the metric configuration
    # The configuration for datasets, collators mentioned in metric config are automatically instantiatied and are provided in kwargs
    tokenizer = kwargs["tokenizer"]
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    generation_args = kwargs["generation_args"]
    ...
    return {
        "agg_value": np.mean(rouge_values),
        "value_by_index": scores_by_index,
    }

# in src/evals/metrics/privacy.py
@unlearning_metric(name="privleak")
def privleak(model, **kwargs):
  # the privleak quality metric is found from computed statistics of 
  # other metrics like MIA attack scores, which is provided through kwargs
  ...
  return {'agg_value': (score-ref)/(ref+1e-10)*100}

```
- `@unlearning_metric(name="rouge")` - Defines a `rouge` handler.

> [!NOTE]
`kwargs` contains many important attributes that are useful while computing metrics. It will contain all the metric-specific parameters defined in the metric's yaml file, and also contain the created objects corresponding to the other attributes mentioned in the metric config: such as the `"tokenizer"`, `"data"` (the preprocessed torch dataset), `"batch_size"`, `"collator"`, `"generation_args"`, `"pre_compute"` (prior metrics the current metric depends on), and `"reference_logs"` (evals from a reference model the current metric can use).

#### retain_logs_path (for privleak and forget_quality)

The metrics **privleak** and **forget_quality** compare the unlearned model to a **retain (baseline) model**. They require pre-computed evals from the retain model, loaded from a JSON file.

**What it means:** The retain model is the model *before* unlearning—typically finetuned on the retain split. Its evals (e.g. MIA AUC for privleak, Truth Ratio for forget_quality) serve as the reference. Without them, privleak falls back to `ref_value` (0.5) and forget_quality returns None.

**How to set it:**

1. **Evaluate your retain model** and save results to JSON:
   ```bash
   python src/eval.py experiment=eval/tofu/default.yaml \
     model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_retain95 \
     paths.output_dir=saves/eval/tofu_retain95
   ```
   This produces `saves/eval/tofu_retain95/TOFU_EVAL.json`.

2. **Pass the path** via Hydra override when evaluating the unlearned model:
   ```bash
   # TOFU (standard or trajectory)
   eval.tofu.retain_logs_path=saves/eval/tofu_retain95/TOFU_EVAL.json
   eval.tofu_trajectory.retain_logs_path=saves/eval/tofu_retain95/TOFU_EVAL.json

   # MUSE
   eval.muse.retain_logs_path=saves/eval/muse_retain/MUSE_EVAL.json
   eval.muse_trajectory.retain_logs_path=saves/eval/muse_retain/MUSE_EVAL.json
   ```

3. **When retain_logs_path is None:** The evaluator logs a warning at the start of the run, at the start of evaluation, and when the metric runs. The run continues; privleak uses the default and forget_quality returns None.

#### 2. Register the metric handler
Register the handler to link the class to the configs via the class name in [`METRIC_REGISTRY`](../src/evals/metrics/__init__.py).

Example: Registering the `rouge` handler

```python
from evals.metrics.memorization import rouge
from evals.metrics.privacy import rouge
_register_metric(rouge)
```

#### 3. Add a metric to configs
Metric configurations are in [`configs/eval/tofu_metrics`](../configs/eval/tofu_metrics/) and [`configs/eval/muse_metrics`](../configs/eval/muse_metrics/). These create individual evaluation metrics by providing the handler a specific dataset and other parameters. Multiple metrics may use the same handler.

Example 1: Creating the config for MUSE's `forget_verbmem_ROUGE` ([`configs/eval/muse_metrics/forget_knowmem_ROUGE.yaml`](../configs/eval/muse_metrics/forget_knowmem_ROUGE.yaml)). 



```yaml
# @package eval.muse.metrics.forget_verbmem_ROUGE
# NOTE: the above line is not a comment. See 
# https://hydra.cc/docs/upgrades/0.11_to_1.0/adding_a_package_directive/
# it ensures that the below attributes are found in the config path
# eval.muse.metrics.forget_verbmem_ROUGE in the final config
defaults: # fill up forget_verbmem_ROUGE's inputs' configs
  - ../../data/datasets@datasets: MUSE_forget_verbmem
  - ../../collator@collators: DataCollatorForSupervisedDatasetwithIndex
  - ../../generation@generation_args: default
handler: rouge # the handler we defined above
rouge_type: rougeL_f1
batch_size: 8
# override default parameters
datasets:
  MUSE_forget_verbmem:
    args:
      hf_args:
        path: muse-bench/MUSE-${eval.muse.data_split}
      predict_with_generate: True
collators:
  DataCollatorForSupervisedDataset: 
    args:
      padding_side: left # for generation
generation_args:
  max_new_tokens: 128
```

Example 2: Creating the config for TOFU's `forget_quality` ([`configs/eval/tofu_metrics/forget_quality.yaml`](../configs/eval/tofu_metrics/forget_quality.yaml)).

```yaml
# @package eval.tofu.metrics.forget_quality
defaults:
  - .@pre_compute.forget_truth_ratio: forget_Truth_Ratio

reference_logs:
 # forget quality is computed by comparing truth_ratio 
 # of the given model to a retain model
 # Way to access in metric function: kwargs["reference_logs"]["retain_model_logs"]["retain"]
  retain_model_logs: # name to acess the loaded logs in metric function
    path: ${eval.tofu.retain_logs_path} # path to load the logs
    include: 
      forget_truth_ratio: # keys to include from the logs
        access_key: retain # name of the key to access it inside metric
      
# since the forget_quality metric depends on another metric (truth ratio)
pre_compute:
  forget_truth_ratio:
    access_key: forget

handler: ks_test # the handler with logic that is registered in code 
```


### Designing metrics that depend on other metrics

Some evaluation metrics are designed as transformations of one or more other metrics. 

Examples: 1. TOFU's Truth Ratio uses probability metrics for true and false model responses. 2. MUSE's PrivLeak uses AUC values computed over MinK% probability metrics. 3. TOFU's Model Utility is a harmonic mean of 9 evaluation metrics that measure a model's utility in various ways.

To remove the need for re-computing such metrics, our evaluators support a "precompute" feature, where one can list the metric dependencies in a metric's configs. These parent metrics are then precomputed and saved in the evaluator and provided to the child metric's handler to perform the transformations. The `forget_quality` config example in the previous section illustrates the usage of the "precompute" features. 

Another example of declaring dependent metrics is `truth_ratio`:

```yaml
# @package eval.tofu.metrics.forget_truth_ratio

defaults: # load parent metric configs under the precompute attribute
  - .@pre_compute.forget_Q_A_PARA_Prob: forget_Q_A_PARA_Prob
  - .@pre_compute.forget_Q_A_PERT_Prob: forget_Q_A_PERT_Prob

pre_compute:
  forget_Q_A_PARA_Prob: # parent metric
    access_key: correct # sets a key to access the pre-computed values from
  forget_Q_A_PERT_Prob: # parent metric
    access_key: wrong

handler: forget_truth_ratio
```

The corresponding handler:
```python
# in src/evals/metrics/memorization.py
@unlearning_metric(name="truth_ratio")
def truth_ratio(model, **kwargs):
    """Compute the truth ratio, aggregating false/true scores, and
    return the aggregated value."""
    # kwargs contains all necessary data, including pre-computes
    ...
    # access pre-computes using the defined access keys
    correct_answer_results = kwargs["pre_compute"]["correct"]["value_by_index"]
    wrong_answer_results = kwargs["pre_compute"]["wrong"]["value_by_index"]
    ...
    return {"agg_value": forget_tr_avg, "value_by_index": value_by_index}
```



## Benchmarks

A benchmark (also called evaluator) is a collection of evaluation metrics defined above (e.g. TOFU, MUSE). To add a new benchmark:

### Implement a handler

In the handlers in [`src/evals`](../src/evals/) ([example](../src/evals/tofu.py)), you can add code to: modify the collection, aggregation and reporting of the metrics computed, any pre-eval model preparation etc.

### Register the benchmark handler
Register the benchmark to link the class to the configs via the class name in [`BENCHMARK_REGISTRY`](../src/evals/__init__.py).

Example: Registering TOFU benchmark

```python
from evals.tofu import TOFUEvaluator
_register_benchmark(TOFUEvaluator)
```

### Add to configs
Evaluator config files are in [`configs/eval`](../configs/eval/), e.g [`configs/eval/tofu.yaml`](../configs/eval/tofu.yaml).

Example: TOFU evaluator config file ([`configs/eval/tofu.yaml`](../configs/eval/tofu.yaml))

```yaml
# @package eval.tofu
defaults: # include all the metrics that come under the TOFU evaluator
  - tofu_metrics: # When you import a metric here, its configuration automatically populates the 
  # metrics mapping below, enabled by the @package directive at the top of each metric config file.
    - forget_quality
    - forget_Q_A_Prob
    - forget_Q_A_ROUGE
    - model_utility # populated in the metrics key as metrics.model_utility

handler: TOFUEvaluator
metrics: {} # lists a mapping from each evaluation metric listed above to its config 
output_dir: ${paths.output_dir} # set to default eval directory
forget_split: forget10
```

## lm-evaluation-harness

To evaluate model capabilities after unlearning, we support running [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main) using our custom evaluator: [LMEvalEvaluator](../src/evals/lm_eval.py).
All evaluation tasks should be defined under the  `tasks` in [lm_eval.yaml](../configs/eval/lm_eval.yaml)

```yaml
# @package eval.lm_eval
# NOTE: the above line is not a comment, but sets the package for config. See https://hydra.cc/docs/upgrades/0.11_to_1.0/adding_a_package_directive/

handler: LMEvalEvaluator
output_dir: ${paths.output_dir} # set to default eval directory
overwrite: false

# Define evaluation tasks here
tasks:
  - mmlu
  - wmdp_cyber
  - task: gsm8k
    dataset_path: gsm8k
    # define the entire task config. 
    # ^ Example: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml
    


simple_evaluate_args:
  batch_size: 16
  system_instruction: null
  apply_chat_template: false
```

