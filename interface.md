# Interface from Vlad

https://gist.github.com/vladislavneon/0e107a5ff73074cd3342564e99e30bdd#file-interface-ipynb
Пример того, как делали ребята в домашках

## Run
Your solution must be in the file named main.py. It has to support all of the keys below:

--ds_path: string; path to dataset file in .svm format.

--optimize_method: string; high-level optimization method, will be one of {'gradient', 'newton', 'hfn', 'bfgs', 'lbfgs', 'l1prox', 'sgd'}.

--line_search: string; linear optimization method, will be one of {'golden_search', 'brent', 'armijo', 'wolfe', 'lipschitz'}. Note that you don't have to support a combination of 'newton' nor 'hfn' nor any 'bfgs' optimization method and 'lipschitz' linear search.

--point_distribution: string; initial weights distribution class, will be one of {'uniform', 'gaussian'}. In case of uniform its parameters must be (-1, 1) and in case of gaussian its parameters must be (0, $\sqrt{10}$).

--seed: int; seed for numpy randomness.

--eps: float; epsilon to use in termination condition.

--cg-tolerance-policy: string; optional key for HFN method; conjugate gradients method tolerance choice policy, will be one of {'const', 'sqrtGradNorm', 'gradNorm'}.

--cg-tolerance-eta: float; optional key for HFN method; conjugate gradients method tolerance parameter eta.

--lbfgs-history-size: int; optional for L-BFGS method; number of entries in history.

--l1_lambda: float; optional key for l1-proximal method; l1-regularization coefficient.

--batch_size: int; optional key for SGD method.

--sgd_n_iters: int, optional key for SGD method.

## Result
Your solution must return a JSON-object of the following structure:

```json
{
    'initial_point': np.array, initial weights vector
    'optimal_point': np.array, weights vector after optimization
    'func_value': double, loss function value after optimization
    'gradient_value': double, gradient norm value after optimization
    'oracle_calls': {
        'f': int, number of calls to calculate loss
        'df': int, number of calls to calculate gradient/derivative
        'd2f': int, number of calls to calculate hessian
    },
    'r_k': double, as used in termination condition
    'working_time': double, optimization time in seconds
}
```
