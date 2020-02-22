func_name=(
    linear
    poisson_regression
)

opt=(
    gradient
    newton
    hfn
    adam
    bfgs
    lbfgs
)
for j in "${opt[@]}"; do
    for i in "${func_name[@]}"; do
        PYTHONPATH="." python src/main/main.py --function_name "$i" --optim_method "$j" --use_save_args_settings
    done
done