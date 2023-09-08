import argparse
from itertools import chain

from datasets import load_dataset, load_from_disk
from transformers import LlamaTokenizerFast


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="openwebtext", choices=["openwebtext", "wikitext"])
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--tokenizer-path", type=str)
    parser.add_argument("--out-path", type=str)
    parser.add_argument("--num-workers", type=int, default=16)
    return parser.parse_args()


def build_dataset():
    args = parse_args()

    if args.name == "openwebtext":
        if args.data_path is not None:
            print(f"Loading from file {args.data_path} ...")
            dataset = load_from_disk(args.data_path)
        else:
            dataset = load_dataset(args.name)
        split_dataset = dataset["train"].train_test_split(test_size=0.0005,
                                                          seed=2357,
                                                          shuffle=True,
                                                          load_from_cache_file=False)
        split_dataset['val'] = split_dataset.pop('test')    # rename the test split to val
    elif args.name == "wikitext":
        if args.data_path is not None:
            print(f"Loading from file {args.data_path} ...")
            dataset = load_from_disk(args.data_path)
        else:
            dataset = load_dataset(args.name, "wikitext-103-v1")
        split_dataset = dataset
        split_dataset['val'] = split_dataset.pop('validation')
    else:
        raise ValueError(f"invalid dataset name {args.name}")

    print(f"Loaded dataset:\n{split_dataset}")

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    if args.tokenizer_path is not None:
        tokenizer = LlamaTokenizerFast.from_pretrained(args.tokenizer_path)
    else:
        tokenizer = LlamaTokenizerFast.from_pretrained('hf-internal-testing/llama-tokenizer')
        tokenizer.save_pretrained('./')

    def tokenize(example):
        ids = tokenizer(example['text'])['input_ids']
        ids.append(tokenizer.eos_token_id)
        return {'input_ids': ids}

    tokenized = split_dataset.map(
        tokenize,
        num_proc=args.num_workers,
        remove_columns=['text'],
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    def process(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= args.block_size:
            total_length = (total_length // args.block_size) * args.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i:i + args.block_size] for i in range(0, total_length, args.block_size)
               ] for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # tokenize the dataset
    processed = tokenized.map(
        process,
        batched=True,
        num_proc=args.num_workers,
        load_from_cache_file=False,
        desc=f"Grouping texts in chunks of {args.block_size}",
    )

    print(f"Tokenized dataset:\n{processed}")

    print(f"Saving dataset to {args.out_path} ...")
    processed.save_to_disk(args.out_path)
    print("Dataset saved.")


if __name__ == '__main__':
    build_dataset()
