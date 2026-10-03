import json

def main():
    result_file = "/root/autodl-tmp/partnr-planner-main/test_my_videos/results/qwen_inference_results_204.json"
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    is_correct = []
    Bucket_A = []
    Bucket_B = []
    Bucket_C = []
    for i in range(len(data)):
        if data[i]["is_correct"]:
            is_correct.append(i)
        if "Bucket_A" in data[i]['bucket']:
            Bucket_A.append(i)
        if "Bucket_B" in data[i]['bucket']:
            Bucket_B.append(i)
        if "Bucket_C" in data[i]['bucket']:
            Bucket_C.append(i)
    count_A = 0
    count_B = 0
    count_C = 0
    for i in range(len(data)):
        if i in is_correct and i in Bucket_A:
            count_A += 1
        if i in is_correct and i in Bucket_B:
            count_B += 1
        if i in is_correct and i in Bucket_C:
            count_C += 1
    correct_rate = float(len(is_correct) / len(data))
    bucket_A_correct_rate = float(count_A / len(Bucket_A))
    bucket_B_correct_rate = float(count_B / len(Bucket_B))
    bucket_C_correct_rate = float(count_C / len(Bucket_C))
    print(f"Results:\ncorrect_rate: {correct_rate}\nbucket_A: {bucket_A_correct_rate}\nbucket_B: {bucket_B_correct_rate}\nbucket_C: {bucket_C_correct_rate}")

if __name__ == "__main__":
    main()