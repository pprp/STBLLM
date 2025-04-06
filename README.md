# STBLLM: Pushing the Limit of Post-Training Quantization for LLMs [[PDF]](https://arxiv.org/abs/2408.01803)

Peijie Dong$^{1,\dagger}$, Lujun Li$^{2,\dagger}$, Yuedong Zhong$^{3}$, Dayou Du$^{1}$, Ruibo Fan$^{1}$, Yuhan Chen$^{1}$, Zhenheng Tang$^{1,4}$, Qiang Wang$^{5}$, Wei Xue$^{2}$, Yike Guo$^{2,*}$, Xiaowen Chu$^{1,2*}$

$^{1}$ HKUST(GZ)    $^{2}$ HKUST    $^{3}$ SYSU    $^{4}$ HKBU    $^{5}$ HIT(SZ)


## News

- [2025/4] *STBLLM* source code is open now!

## Dependencies

```
conda create -n stbllm python==3.10
pip install -r requirements.txt
```

* `torch`: tested on v26.0
* `transformers`: tested on v4.35.0
* `datasets`: tested on v2.14.6
* `huggingface-hub`: tested on v0.16.4

## LLMs Binarization

#### Binarization for OPT families

```
python3 run.py facebook/opt-6.7b c4 braq --blocksize 128 --salient_metric hessian
```


#### Binarization for LLaMA families

```
python3 run.py meta-llama/Llama-2-7b-hf c4 braq --blocksize 128 --salient_metric hessian
```
or
```
python3 run.py huggyllama/llama-7b c4 braq --blocksize 128 --salient_metric hessian
```

#### Binarization for Vicuna families (Instruction Fine-tuning Models)

```
python3 run.py lmsys/vicuna-7b-v1.5 c4 braq --blocksize 128 --salient_metric hessian
```

#### 

## Results

- STBLLM  achieve superior perplexity performance on Wikitext2 datasets  within only an average of **1.11** bit-width weights OPT families.

![intuition](imgs/opt_wiki_results.png)

- STBLLM  achieve superior perplexity performance on Wikitext2 datasets  within only an average of **1.09** bit-width weights LLaMA families and **1.08** bit-width weights LLaMA2 families.

![intuition](imgs/llama_wiki_results.png)

- We also evaluated the performance of *STBLLM* on PTB and C4 datasets. 

![intuition](imgs/ptb1.png)

![intuition](imgs/ptb2.png)

- We further evaluated *STBLLM* on 7 zero-shot dataset to give extensive insight on  binarization LLMs

  ![intuition](imgs/zero_shot.png)

- STBLLM  achieve superior perplexity performance on Wikitext2 datasets  within only an average of **1.10** bit-width weights Vicuna families (instruction fine-tune models).

![intuition](imgs/vicuna.png)

## Related Project


[GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers](https://github.com/IST-DASLab/gptq)

[PB-LLM: Partially Binarized Large Language Models](https://github.com/hahnyuan/PB-LLM)

[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://github.com/mit-han-lab/llm-awq)




## Acknowledgements 

SparseGPT:
Wanda:
STBLLM: 
RIA: 
OWL: 

## Citation

If you find *STBLLM* is useful and helpful to your work, please kindly cite this paper:

```bibtex
@inproceedings{dong2025stbllm,
    title={STBLLM: Breaking the 1-Bit Barrier with Structured Binary LLMs},
    author={Peijie Dong and Lujun Li and Yuedong Zhong and DaYou Du and Ruibo FAN and Yuhan Chen and Zhenheng Tang and Qiang Wang and Wei Xue and Yike Guo and Xiaowen Chu},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2025},
    url={https://openreview.net/forum?id=6XUSDvBFkV#}, 
}
```
