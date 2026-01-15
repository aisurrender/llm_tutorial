"""
使用 SentencePiece 训练 BPE 分词器

SentencePiece 是 Google 开发的分词工具，支持 BPE 和 Unigram 算法。
实际项目中，我们通常使用 SentencePiece 而不是自己实现 BPE。

运行: python train_bpe.py --data your_text.txt --vocab_size 6400
"""

import argparse
import os


def train_sentencepiece(data_file: str, model_prefix: str, vocab_size: int):
    """
    使用 SentencePiece 训练分词器

    Args:
        data_file: 训练数据文件路径
        model_prefix: 模型保存前缀
        vocab_size: 词表大小
    """
    try:
        import sentencepiece as spm
    except ImportError:
        print("请先安装 sentencepiece: pip install sentencepiece")
        return

    # 训练参数
    spm.SentencePieceTrainer.train(
        input=data_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',  # 使用 BPE 算法
        character_coverage=0.9995,  # 字符覆盖率（中文建议 0.9995）
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )

    print(f"模型已保存: {model_prefix}.model, {model_prefix}.vocab")


def demo_sentencepiece(model_path: str):
    """演示 SentencePiece 分词器的使用"""
    try:
        import sentencepiece as spm
    except ImportError:
        print("请先安装 sentencepiece: pip install sentencepiece")
        return

    # 加载模型
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    print(f"\n词表大小: {sp.get_piece_size()}")

    # 测试编码解码
    test_texts = [
        "Hello world!",
        "你好世界！",
        "The quick brown fox jumps over the lazy dog.",
        "机器学习是人工智能的一个分支。",
    ]

    print("\n编码解码测试:")
    for text in test_texts:
        # 编码为 token IDs
        ids = sp.encode(text)
        # 编码为 token 字符串
        pieces = sp.encode_as_pieces(text)
        # 解码
        decoded = sp.decode(ids)

        print(f"\n原文: {text}")
        print(f"Tokens: {pieces}")
        print(f"IDs: {ids}")
        print(f"解码: {decoded}")


def create_sample_data(output_file: str):
    """创建示例训练数据"""
    sample_text = """
The quick brown fox jumps over the lazy dog.
Machine learning is a branch of artificial intelligence.
Deep learning models can process large amounts of data.
Natural language processing enables computers to understand human language.
Transformers have revolutionized the field of NLP.
GPT models are based on the transformer architecture.
Large language models can generate human-like text.
Pre-training and fine-tuning are key techniques in modern NLP.
机器学习是人工智能的一个重要分支。
深度学习模型可以处理大量数据。
自然语言处理使计算机能够理解人类语言。
Transformer架构革新了自然语言处理领域。
GPT模型基于Transformer架构。
大型语言模型可以生成类似人类的文本。
预训练和微调是现代NLP的关键技术。
你好世界，这是一个测试文本。
人工智能正在改变我们的生活。
""" * 10  # 重复多次以增加数据量

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(sample_text)

    print(f"示例数据已保存到: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 BPE 分词器")
    parser.add_argument("--data", type=str, default="sample_data.txt",
                       help="训练数据文件路径")
    parser.add_argument("--model_prefix", type=str, default="tokenizer",
                       help="模型保存前缀")
    parser.add_argument("--vocab_size", type=int, default=1000,
                       help="词表大小")
    parser.add_argument("--demo", action="store_true",
                       help="仅演示已训练的模型")
    parser.add_argument("--create_sample", action="store_true",
                       help="创建示例训练数据")

    args = parser.parse_args()

    if args.create_sample:
        create_sample_data(args.data)

    if args.demo:
        model_path = f"{args.model_prefix}.model"
        if os.path.exists(model_path):
            demo_sentencepiece(model_path)
        else:
            print(f"模型文件不存在: {model_path}")
            print("请先训练模型: python train_bpe.py --data your_data.txt")
    else:
        if not os.path.exists(args.data):
            print(f"数据文件不存在: {args.data}")
            print("创建示例数据: python train_bpe.py --create_sample")
            print("然后训练: python train_bpe.py --data sample_data.txt")
        else:
            train_sentencepiece(args.data, args.model_prefix, args.vocab_size)
            demo_sentencepiece(f"{args.model_prefix}.model")
