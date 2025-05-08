MODEL="Llama-3.1-8B-Instruct"

#VLLM_WORKER_MULTIPROC_METHOD="spawn" python eval.py --data results/Llama-3.1-8B-Instruct_thresh_w64_p0.5_maxlen4000.json --csv results/accuracy_Llama3.1_8B_Instruct.csv --gpu 4
#VLLM_WORKER_MULTIPROC_METHOD="spawn" python eval.py --data results/Llama-3.1-8B-Instruct_thresh_w64_p0.75_maxlen4000.json --csv results/accuracy_Llama3.1_8B_Instruct.csv --gpu 4 
VLLM_WORKER_MULTIPROC_METHOD="spawn" python eval.py --data results/Llama-3.1-8B-Instruct_thresh_w64_p0.875_maxlen4000.json --csv results/accuracy_Llama3.1_8B_Instruct.csv --gpu 4 
#VLLM_WORKER_MULTIPROC_METHOD="spawn" python eval.py --data results/Llama-3.1-8B-Instruct_thresh_w128_p0.5_maxlen4000.json --csv results/accuracy_Llama3.1_8B_Instruct.csv --gpu 4
#VLLM_WORKER_MULTIPROC_METHOD="spawn" python eval.py --data results/Llama-3.1-8B-Instruct_thresh_w128_p0.75_maxlen4000.json --csv results/accuracy_Llama3.1_8B_Instruct.csv --gpu 4
#VLLM_WORKER_MULTIPROC_METHOD="spawn" python eval.py --data results/Llama-3.1-8B-Instruct_thresh_w128_p0.875_maxlen4000.json --csv results/accuracy_Llama3.1_8B_Instruct.csv --gpu 4
#VLLM_WORKER_MULTIPROC_METHOD="spawn" python eval.py --data results/Llama-3.1-8B-Instruct_thresh_w256_p0.5_maxlen4000.json --csv results/accuracy_Llama3.1_8B_Instruct.csv --gpu 4
#VLLM_WORKER_MULTIPROC_METHOD="spawn" python eval.py --data results/Llama-3.1-8B-Instruct_thresh_w256_p0.75_maxlen4000.json --csv results/accuracy_Llama3.1_8B_Instruct.csv --gpu 4
#VLLM_WORKER_MULTIPROC_METHOD="spawn" python eval.py --data results/Llama-3.1-8B-Instruct_thresh_w256_p0.875_maxlen4000.json --csv results/accuracy_Llama3.1_8B_Instruct.csv --gpu 4
#VLLM_WORKER_MULTIPROC_METHOD="spawn" python eval.py --data results/Llama-3.1-8B-Instruct_thresh_w512_p0.5_maxlen4000.json --csv results/accuracy_Llama3.1_8B_Instruct.csv --gpu 4
#VLLM_WORKER_MULTIPROC_METHOD="spawn" python eval.py --data results/Llama-3.1-8B-Instruct_thresh_w512_p0.75_maxlen4000.json --csv results/accuracy_Llama3.1_8B_Instruct.csv --gpu 4
#VLLM_WORKER_MULTIPROC_METHOD="spawn" python eval.py --data results/Llama-3.1-8B-Instruct_thresh_w512_p0.875_maxlen4000.json --csv results/accuracy_Llama3.1_8B_Instruct.csv --gpu 4
#VLLM_WORKER_MULTIPROC_METHOD="spawn" python eval.py --data results/Llama-3.1-8B-Instruct_flash_w0_p0_maxlen4000.json --csv results/accuracy_Llama3.1_8B_Instruct.csv --gpu 4
