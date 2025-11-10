# To be executed as `. run.sh` (or `source run.sh`)

micromamba activate -n pbi

classifiers=(
    # '{"name":"LinearClassifier"}'
    '{"name":"CNNClassifier","params":{"bacterium_conv_params":"[(64,3,5),(32,10,5)]","phage_conv_params":"[(64,3,5),(32,10,5)]","dense_dim":100,"dense_dropout":0.5}}'
    '{"name":"CNNClassifier","params":{"bacterium_conv_params":"[(64,3,5),(32,10,5),(32,10,5)]","phage_conv_params":"[(64,3,5),(32,10,5)]","dense_dim":100,"dense_dropout":0.5}}'
    '{"name":"CNNClassifier","params":{"bacterium_conv_params":"[(128,3,5),(32,10,5),(32,10,5)]","phage_conv_params":"[(64,3,5),(32,10,5)]","dense_dim":100,"dense_dropout":0.5}}'
    '{"name":"CNNClassifier","params":{"bacterium_conv_params":"[(256,3,5),(32,10,5),(32,10,5)]","phage_conv_params":"[(64,3,5),(32,10,5)]","dense_dim":100,"dense_dropout":0.5}}'
    '{"name":"CNNClassifier","params":{"bacterium_conv_params":"[(512,3,5),(32,10,5),(32,10,5)]","phage_conv_params":"[(64,3,5),(32,10,5)]","dense_dim":100,"dense_dropout":0.5}}'
    # '{"name":"CNNClassifier","params":{"bacterium_conv_params":"[(256,3,5),(32,3,5),(32,10,5)]","phage_conv_params":"[(64,3,5),(32,10,5)]","dense_dim":100,"dense_dropout":0.5}}'
    # '{"name":"CNNClassifier","params":{"bacterium_conv_params":"[(256,3,5),(64,3,5),(32,10,5)]","phage_conv_params":"[(64,3,5),(32,10,5)]","dense_dim":100,"dense_dropout":0.5}}'
    # '{"name":"CNNClassifier","params":{"bacterium_conv_params":"[(256,3,5),(64,10,5),(32,10,5)]","phage_conv_params":"[(64,3,5),(32,10,5)]","dense_dim":100,"dense_dropout":0.5}}'
    # '{"name":"CNNClassifier","params":{"bacterium_conv_params":"[(256,3,5),(128,3,5),(32,10,5)]","phage_conv_params":"[(64,3,5),(32,10,5)]","dense_dim":100,"dense_dropout":0.5}}'
    # '{"name":"CNNClassifier","params":{"bacterium_conv_params":"[(256,3,5),(128,10,5),(32,10,5)]","phage_conv_params":"[(64,3,5),(32,10,5)]","dense_dim":100,"dense_dropout":0.5}}'
    # '{"name":"CNNClassifier","params":{"bacterium_conv_params":"[(256,3,5),(64,10,5),(16,10,5)]","phage_conv_params":"[(64,3,5),(32,10,5)]","dense_dim":100,"dense_dropout":0.5}}'
    # '{"name":"CNNClassifier","params":{"bacterium_conv_params":"[(256,3,5),(128,10,5),(16,10,5)]","phage_conv_params":"[(64,3,5),(32,10,5)]","dense_dim":100,"dense_dropout":0.5}}'
)

echo "Classifiers: ${classifiers[@]}"

for classifier in "${classifiers[@]}"; do
    echo "==============================================================================================================="
    echo "CLASSIFIER: ${classifier}"
    echo "==============================================================================================================="

    CLASSIFIER=$classifier python main.py -c model_configs/temporal_best_env.yaml
done