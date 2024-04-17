#python3 run.py facebook/opt-6.7b c4 braq --blocksize 128 --salient_metric hessian --device "cuda:0"
# python3 run.py /home/dongpeijie/share/llama-2-7b wikitext2 braq --blocksize 128 --salient_metric hessian --device "cuda:0"

python3 run.py /data2/share/llama-1/llama-7b-hf wikitext2 braq --blocksize 128 --salient_metric hessian --device "cuda:0"
