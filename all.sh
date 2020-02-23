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
    l1prox
)
for j in "${opt[@]}"; do
    for i in "${func_name[@]}"; do
        PYTHONPATH="." python src/main/main.py --function_name "$i" --optim_method "$j" --use_save_args_settings
        echo -e "\n\n"
    done
done