import os
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from tqdm import tqdm
import pdb
import numpy as np
import shutil

birefnet = AutoModelForImageSegmentation.from_pretrained("/data1/sjj/Troika/pretrained_checkpoint/ZhengPeng7/BiRefNet", trust_remote_code=True)

torch.set_float32_matmul_precision(['high', 'highest'][0])
birefnet.to('cuda')
birefnet.eval()

def extract_object(birefnet, imagepath, output_path):
    # Data settings
    image_size = (512, 512)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(imagepath)
    input_images = transform_image(image).unsqueeze(0).to('cuda')

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)

    # # 透明背景 ###
    # image_with_mask = Image.composite(image, Image.new("RGB", image.size, (0, 0, 0)), mask)

    ### 全白色背景 #####
    mask = mask.point(lambda p: 255 if p > 128 else 0)
    # Apply mask to the image to make non-target areas white
    image_array = np.array(image)
    mask_array = np.array(mask)
    # Create a white background where the mask is 0
    image_array[mask_array == 0] = [255, 255, 255]
    # Convert back to PIL Image
    image_with_mask = Image.fromarray(image_array)



    white_pixels = np.sum(np.all(image_array == [255, 255, 255], axis=-1))
    total_pixels = image_array.shape[0] * image_array.shape[1]
    white_percentage = white_pixels / total_pixels

    if white_percentage > 0.95:
        # Save the original image if condition is met
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)

    # Ensure the output directory exists
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image_with_mask.save(output_path)



# # Function to process all images in a directory and rename the largest folder
def process_all_images(root_folder):
    output_base_folder = os.path.join(root_folder, 'images_woback')

    # Use tqdm to add a progress bar
    for dirpath, dirnames, filenames in tqdm(os.walk(root_folder), desc="Processing folders"):
        relative_path = os.path.relpath(dirpath, root_folder)
        output_folder = os.path.join(output_base_folder, relative_path)
        image_count = 0
        
        for filename in filenames:
            if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff')):
                image_path = os.path.join(dirpath, filename)
                output_path = os.path.join(output_folder.replace('/images', ''), filename)
                # target = os.path.dirname(output_path)
                # pdb.set_trace()
                try:
                    extract_object(birefnet, image_path, output_path)
                    image_count += 1
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    # Copy the original image to the output path
                    shutil.copy(image_path, os.path.dirname(output_path))
                    # pdb.set_trace()

# Main code to run the function
if __name__ == "__main__":
    folder_path = '/data1/sjj/Troika/dataset_root/ut-zap50k/images'
    process_all_images(folder_path)








# ##### 原始 #####

# import os
# from PIL import Image
# import torch
# import torchvision.transforms as transforms
# import depth_pro
# from tqdm import tqdm
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm

# # Load model and preprocessing transform
# model, transform = depth_pro.create_model_and_transforms()
# model.eval()

# # 定义输入和输出文件夹路径
# input_folder = '/data1/sjj/Troika/dataset_root/food101/images/Bake_Bread'
# output_folder = '/data1/sjj/Troika/dataset_root/processed_depth_images'

# # 确保输出文件夹存在
# os.makedirs(output_folder, exist_ok=True)

# # 获取输入文件夹中的所有jpg文件
# jpg_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

# # 遍历输入文件夹中的所有jpg文件，使用tqdm显示进度条
# for filename in tqdm(jpg_files, desc="Processing Images"):
#     input_path = os.path.join(input_folder, filename)
    
#     # Load and preprocess an image
#     image, _, f_px = depth_pro.load_rgb(input_path)
#     image = transform(image)
    
#     # Run inference
#     prediction = model.infer(image, f_px=f_px)
#     depth = prediction["depth"]  # Depth in [m]

#     # 将深度图像转换为NumPy数组
#     depth_np = depth.cpu().numpy().squeeze()
    
#     # 归一化深度图，以便颜色映射
#     depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
    
#     depth_normalized = 1- depth_normalized
#     # 应用伪彩色映射
#     colormap = cm.jet(depth_normalized)  # 使用jet色图
#     depth_color_mapped = (colormap[:, :, :3] * 255).astype(np.uint8)  # 转换为RGB格式

#     # 转换为PIL图像并保存
#     depth_image = Image.fromarray(depth_color_mapped)
#     output_path = os.path.join(output_folder, f'depth_{filename}')
#     depth_image.save(output_path)

# print("所有图片处理完成并保存到新的文件夹中。")

# # print(depth.shape)
# # focallength_px = prediction["focallength_px"]  # Focal length in pixels.
# # print(focallength_px)

#####


#### 新的 #####

# import os
# from PIL import Image
# import torch
# import torchvision.transforms as transforms
# import depth_pro
# from tqdm import tqdm
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm

# # 加载模型和预处理转换
# model, transform = depth_pro.create_model_and_transforms()
# model.eval()

# # 定义输入和输出文件夹路径
# input_root = '/data1/sjj/Troika/dataset_root/ut-zap50k/images'
# output_root = '/data1/sjj/Troika/dataset_root/ut-zap50k/images_depth'

# # 确保输出根文件夹存在
# os.makedirs(output_root, exist_ok=True)

# # 计算总的图片数量以显示总体进度条
# total_images = sum(len(files) for _, _, files in os.walk(input_root) if files)

# # 外层总体进度条
# with tqdm(total=total_images, desc="Overall Progress") as pbar:
#     # 遍历输入文件夹下的所有子文件夹
#     for subdir in os.listdir(input_root):
#         subfolder_path = os.path.join(input_root, subdir)
#         if os.path.isdir(subfolder_path):  # 仅处理子文件夹
#             output_subfolder = os.path.join(output_root, subdir)
#             os.makedirs(output_subfolder, exist_ok=True)  # 确保输出子文件夹存在
            
#             # 获取子文件夹中的所有jpg文件
#             jpg_files = [f for f in os.listdir(subfolder_path) if f.endswith('.jpg')]
            
#             # 遍历子文件夹中的所有jpg文件
#             for filename in jpg_files:
#                 input_path = os.path.join(subfolder_path, filename)
                
#                 # 加载并预处理图像
#                 image, _, f_px = depth_pro.load_rgb(input_path)
#                 image = transform(image)
                
#                 # 推理
#                 prediction = model.infer(image, f_px=f_px)
#                 depth = prediction["depth"]  # 深度信息
                
#                 # 将深度图像转换为NumPy数组
#                 depth_np = depth.cpu().numpy().squeeze()
                
#                 # 归一化深度图，用于颜色映射
#                 depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
#                 depth_normalized = 1 - depth_normalized  # 反转深度图
                
#                 # 应用伪彩色映射
#                 colormap = cm.jet(depth_normalized)
#                 depth_color_mapped = (colormap[:, :, :3] * 255).astype(np.uint8)  # 转为RGB格式
                
#                 # 转换为PIL图像并保存
#                 depth_image = Image.fromarray(depth_color_mapped)
#                 output_path = os.path.join(output_subfolder, f'{filename}')
#                 depth_image.save(output_path)
                
#                 # 更新总体进度条
#                 pbar.update(1)

# print("所有图片处理完成并按子文件夹结构保存到新的文件夹中。")









# import os

# def count_files_and_folders(path):
#     folder_count = 0
#     file_count = 0
    
#     for root, dirs, files in os.walk(path):
#         folder_count += len(dirs)
#         file_count += len(files)
    
#     return folder_count, file_count

# # 输入你的文件夹路径
# path = '/data1/sjj/Troika/dataset_root/mit-states/images'
# folders, files = count_files_and_folders(path)

# print(f'子文件夹数量: {folders}')
# print(f'文件数量: {files}')







# import os

# def list_all_files(folder):
#     """
#     列出文件夹下所有文件的相对路径
#     """
#     file_list = []
#     for dirpath, dirnames, filenames in os.walk(folder):
#         for filename in filenames:
#             relative_path = os.path.relpath(os.path.join(dirpath, filename), folder)
#             file_list.append(relative_path)
#     return set(file_list)

# def compare_folders(folder1, folder2):
#     """
#     比较两个文件夹，找出差异
#     """
#     files_in_folder1 = list_all_files(folder1)
#     files_in_folder2 = list_all_files(folder2)

#     only_in_folder1 = files_in_folder1 - files_in_folder2
#     only_in_folder2 = files_in_folder2 - files_in_folder1

#     return only_in_folder1, only_in_folder2

# # 输入两个文件夹路径
# folder1 = '/data1/sjj/Troika/dataset_root/mit-states/images'
# folder2 = '/data1/sjj/Troika/dataset_root/mit-states/images_noback'

# only_in_folder1, only_in_folder2 = compare_folders(folder1, folder2)

# # 输出结果
# print(f"Only in {folder1} ({len(only_in_folder1)} files):")
# for file in sorted(only_in_folder1):
#     print(file)

# print(f"\nOnly in {folder2} ({len(only_in_folder2)} files):")
# for file in sorted(only_in_folder2):
#     print(file)