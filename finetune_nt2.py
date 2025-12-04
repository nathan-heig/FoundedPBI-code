import os
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer # type: ignore
from transformers.models.bert.configuration_bert import BertConfig
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
import matplotlib.pyplot as plt
import math
import argparse
from pbi_utils.logging import Logging, DEBUG

logger = Logging()
logger.set_logging_level(DEBUG)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a masked language model with LoRA")
    parser.add_argument("--dataset_base_dir", type=str, required=True, help="Base folder of the datasets")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (folder name inside `dataset_base_dir`)")
    parser.add_argument("--base_model_name", type=str, required=True, help="Base pretrained model name or path, which will be fine-tuned")
    parser.add_argument("--output_model_dir", type=str, required=True, help="Output folder to save the fine-tuned model")
    parser.add_argument("--output_model_name", type=str, required=True, help="Output model name (folder name inside `output_model_dir`)")
    parser.add_argument("--output_plots_dir", type=str, required=False, default="auto", help="Output folder to save training plots. If 'auto', plots will be saved inside `output_model_dir`/`output_model_name`/plots")
    
    parser.add_argument("--model_type", type=str, choices=["nt2", "dnabert2"], default="nt2", help="Type of model to fine-tune: 'nt2' for Nucleotide Transformer 2, 'dnabert2' for DNABERT2")

    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated list of GPU IDs to use for training")
    parser.add_argument("--num_proc", type=int, default=16, help="Number of processes for dataset tokenization")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length for tokenization")

    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")

    parser.add_argument("--per_device_batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training")
    parser.add_argument("--no_use_bf16", action="store_true", default=False, help="Disable bf16 training, use fp16 instead")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=False, help="Resume training from the last checkpoint")

    cli_args = parser.parse_args()

    num_gpus = len(cli_args.gpu_ids.split(","))
    if num_gpus > 1:
        logger.error("Multi-GPU finetuning will probably not train properly, continue at your own risk.")

    if cli_args.output_plots_dir == "auto":
        cli_args.output_plots_dir = os.path.join(cli_args.output_model_dir, cli_args.output_model_name, "plots")

    # Log CLI arguments
    logger.debug("CLI Arguments:")
    for arg, value in vars(cli_args).items():
        logger.debug(f"  {arg}: {value}")
    


    return cli_args

def config_environment(cli_args: argparse.Namespace):
    os.environ["CUDA_VISIBLE_DEVICES"] = cli_args.gpu_ids
    os.makedirs(cli_args.output_model_dir, exist_ok=True)
    os.makedirs(cli_args.output_plots_dir, exist_ok=True)

def get_last_checkpoint(output_dir: str) -> str | None:
    last_checkpoint = None
    if os.path.isdir(output_dir):
        checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            last_checkpoint = max(checkpoints, key=lambda x: int(x.rsplit("-", 1)[-1]))
    return last_checkpoint

def load_transformers_model_nt2(cli_args: argparse.Namespace):
    # if cli_args.resume_from_checkpoint:
    #     # Find last checkpoint
    #     last_checkpoint = get_last_checkpoint(os.path.join(cli_args.output_model_dir, cli_args.output_model_name))
    #     if last_checkpoint:
    #         logger.info(f"Loading model from last checkpoint: {last_checkpoint}")
    #         base_model = AutoModelForMaskedLM.from_pretrained(cli_args.base_model_name,trust_remote_code=True)
    #         model = PeftModel.from_pretrained(base_model, last_checkpoint, is_trainable=True)
    #         tokenizer = AutoTokenizer.from_pretrained(cli_args.base_model_name)
            
    #         return model, tokenizer
        
    #     logger.info("No checkpoint found, loading base model.")

    model = AutoModelForMaskedLM.from_pretrained(cli_args.base_model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(cli_args.base_model_name)

    lora_config = LoraConfig(
        r=cli_args.lora_rank,
        lora_alpha=cli_args.lora_alpha,
        lora_dropout=cli_args.lora_dropout,
        target_modules=["query", "value"],
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)

    return model, tokenizer

def load_transformers_model_dnabert2(cli_args: argparse.Namespace):
    config = BertConfig.from_pretrained(cli_args.base_model_name)
    model = AutoModel.from_pretrained(cli_args.base_model_name, trust_remote_code=True, config=config)
    tokenizer = AutoTokenizer.from_pretrained(cli_args.base_model_name, trust_remote_code=True)

    lora_config = LoraConfig(
        r=cli_args.lora_rank,
        lora_alpha=cli_args.lora_alpha,
        lora_dropout=cli_args.lora_dropout,
        target_modules=["query", "value", "key", "dense"],
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)

    return model, tokenizer

def load_dataset(cli_args: argparse.Namespace, tokenizer, model_max_length: int):
    dataset_path = os.path.join(cli_args.dataset_base_dir, cli_args.dataset_name)
    dataset = Dataset.load_from_disk(dataset_path)

    def tokenize(batch):
        return tokenizer(
            batch["sequence"],
            truncation=True,
            max_length=model_max_length,
        )
    
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["sequence"], num_proc=cli_args.num_proc).train_test_split(test_size=0.1, seed=42)
    return tokenized

def setup_finetune(cli_args: argparse.Namespace, model, tokenizer, tokenized_dataset):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(cli_args.output_model_dir, cli_args.output_model_name),
        overwrite_output_dir=True,
        eval_strategy="steps",
        logging_strategy="steps",
        per_device_train_batch_size=cli_args.per_device_batch_size,
        per_device_eval_batch_size=cli_args.per_device_batch_size*2,
        gradient_accumulation_steps=4,
        learning_rate=cli_args.learning_rate,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=cli_args.no_use_bf16,
        bf16=not cli_args.no_use_bf16,
        save_steps=500,
        logging_steps=500,
        num_train_epochs=cli_args.epochs,
        # eval_on_start=True,
        # torch_empty_cache_steps=200,
)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )

    # last_checkpoint = get_last_checkpoint(os.path.join(cli_args.output_model_dir, cli_args.output_model_name))
    # if last_checkpoint:
    #     trainer.state.global_step = int(last_checkpoint.rsplit("-", 1)[-1])
    #     trainer.state.epoch = trainer.state.global_step / len(trainer.get_train_dataloader()) * trainer.args.train_batch_size
    #     logger.info(f"Resuming trainer state: global_step={trainer.state.global_step}, epoch={trainer.state.epoch:.2f}")

    return trainer

def plot_training_logs(log_history: list, output_path: str):

    loss =[[a['step'],a['eval_loss']] for a in log_history if 'eval_loss' in a.keys()]
    eval_loss = [c[1] for c in loss]
    eval_steps = [c[0] for c in loss]

    loss =[[a['step'],a['loss']] for a in log_history if 'loss' in a.keys()]
    train_loss = [c[1] for c in loss]
    train_steps = [c[0] for c in loss]

    plt.figure(figsize=(10, 5))
    plt.plot(train_steps, train_loss, label='Training Loss')
    plt.plot(eval_steps, eval_loss, label='Evaluation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss over Steps')
    plt.legend()
    # plt.grid()
    plt.savefig(output_path)

if __name__ == "__main__":
    cli_args = parse_args()

    config_environment(cli_args)

    if cli_args.model_type == "nt2":
        model, tokenizer = load_transformers_model_nt2(cli_args)
    elif cli_args.model_type == "dnabert2":
        model, tokenizer = load_transformers_model_dnabert2(cli_args)

    tokenized_dataset = load_dataset(cli_args, tokenizer, cli_args.max_length)

    trainer = setup_finetune(cli_args, model, tokenizer, tokenized_dataset)

    results = trainer.evaluate()
    logger.info(f"Initial Perplexity: {math.exp(results['eval_loss']):.2f}")

    trainer.train(resume_from_checkpoint=cli_args.resume_from_checkpoint)

    results = trainer.evaluate()
    logger.info(f"Final Perplexity: {math.exp(results['eval_loss']):.2f}")

    trainer.save_model(os.path.join(cli_args.output_model_dir, cli_args.output_model_name))

    plot_path = os.path.join(cli_args.output_plots_dir, f"training_loss_epoch{cli_args.epochs}.png")
    plot_training_logs(trainer.state.log_history, plot_path)
    logger.info(f"Training loss plot saved to {plot_path}")



