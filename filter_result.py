import os
import pprint

# 指定包含.log文件的文件夹路径
folder_path = "./logs/search"


NUM_TOTAL_FILES = 0
NUM_FILTERED_FILES = 0
extracted_g2p = {}
# graph to ppl

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith(".log"):  # 检查文件扩展名是否为.log
        file_path = os.path.join(folder_path, filename)  # 获取文件的完整路径
        NUM_TOTAL_FILES += 1
        with open(file_path, "r") as file:  # 打开文件进行读取
            lines = file.readlines()  # 读取文件的最后一行
            last_line = lines[-1]
            if len(lines) < 200:
                os.remove(file_path)
                continue 
            
            if last_line.startswith(
                "UnboundLocalError"
            ) or last_line.startswith("RuntimeError"):
                # 检查最后一行是否以RuntimeError开头
                print(f"Filtered out: {filename}")  # 打印出需要过滤掉的文件名
                # 这里可以添加删除文件的代码，例如：
                os.remove(file_path)  # 谨慎使用，这将删除文件
                NUM_FILTERED_FILES += 1
                
            
                
            if last_line.startswith(
                "Perplexity"
            ):  # 检查最后一行是否以特定字符串开头
                try:
                    # 提取数字，这里假设数字紧跟在字符串后面，并且是一个浮点数
                    number = float(last_line.split()[-1])
                    # if str(number).startswith("5.47"):
                    #     continue
                    # if str(number).startswith("nan"):
                    #     continue
                    # if number > 6.8:
                    #     continue
                    # if number < 5.8:
                    #     os.remove(file_path)
                    #     NUM_FILTERED_FILES += 1
                    #     continue
                    for line in lines:
                        # Current Graph String is:  W:(MEAN,STANDARDIZE,Z_SCALE)-X[ROW]:(MEAN)
                        if line.startswith("Current graph:"):
                            k = line.split("is:")[-1].strip()
                            extracted_g2p[k] = number

                except ValueError:
                    # 如果转换失败（例如，字符串中不包含有效的浮点数），则打印错误信息
                    print(f"Error extracting number from the last line of {filename}")


print(f"Total files: {NUM_TOTAL_FILES}")
print(f"Filtered files: {NUM_FILTERED_FILES}")
print(f"Files remaining: {NUM_TOTAL_FILES - NUM_FILTERED_FILES}")

pprint.pprint(extracted_g2p)

