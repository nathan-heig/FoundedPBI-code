#!/usr/bin/env bash
# To be executed as `. run.sh` (or `source run.sh`)

micromamba activate -n pbi

classifiers=(
    # '{"name":"LinearClassifier"}'
    # '{"name":"MLPClassifier","params":{"bacterium_mlp_sizes":[128,64],"phage_mlp_sizes":[256,128,64],"dense_dim":100,"dropout":0.2}}'
    # '{"name":"MLPClassifier","params":{"bacterium_mlp_sizes":[256,128,64],"phage_mlp_sizes":[256,128,64],"dense_dim":100,"dropout":0.2}}'
    # '{"name":"MLPClassifier","params":{"bacterium_mlp_sizes":[512,256,128,64],"phage_mlp_sizes":[256,128,64],"dense_dim":100,"dropout":0.2}}'
    # '{"name":"MLPClassifier","params":{"bacterium_mlp_sizes":[128,64],"phage_mlp_sizes":[512,256,128,64],"dense_dim":100,"dropout":0.2}}'
    # '{"name":"MLPClassifier","params":{"bacterium_mlp_sizes":[128,64],"phage_mlp_sizes":[128,64],"dense_dim":100,"dropout":0.2}}'
    # '{"name":"MLPClassifier","params":{"bacterium_mlp_sizes":[128,64],"phage_mlp_sizes":[128,64],"dense_dim":128,"dropout":0.2}}'
    # '{"name":"MLPClassifier","params":{"bacterium_mlp_sizes":[128,64],"phage_mlp_sizes":[128,64],"dense_dim":64,"dropout":0.2}}'
    # '{"name":"MLPClassifier","params":{"bacterium_mlp_sizes":[128,64],"phage_mlp_sizes":[128,64],"dense_dim":256,"dropout":0.2}}'
    # '{"name":"BasicMLPClassifier","params":{"mlp_params":[64],"dropout":0.2}}'
    # '{"name":"BasicMLPClassifier","params":{"mlp_params":[128,64],"dropout":0.2}}'
    # '{"name":"BasicMLPClassifier","params":{"mlp_params":[256,128,64],"dropout":0.2}}'
    # '{"name":"BasicMLPClassifier","params":{"mlp_params":[512,256,128,64],"dropout":0.2}}'
    # '{"name":"BasicMLPClassifier","params":{"mlp_params":[1024,512,256,128,64],"dropout":0.2}}'
    '{"name":"BasicMLPClassifier","params":{"mlp_params":[256,128],"dropout":0.2}}'
    '{"name":"BasicMLPClassifier","params":{"mlp_params":[512,64],"dropout":0.2}}'
    '{"name":"BasicMLPClassifier","params":{"mlp_params":[512,128],"dropout":0.2}}'
    '{"name":"BasicMLPClassifier","params":{"mlp_params":[1204,128],"dropout":0.2}}'
    )

echo "Classifiers: ${classifiers[@]}"

for classifier in "${classifiers[@]}"; do
    echo "==============================================================================================================="
    echo "CLASSIFIER: ${classifier}"
    echo "==============================================================================================================="

    CLASSIFIER=$classifier GPU_ID=3 python main.py -c model_configs/temporal_best_env.yaml
done