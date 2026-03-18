from PIL import Image
import os

def convert_and_rename_images(input_folder, output_folder):
    # 確保輸出資料夾存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 獲取輸入資料夾中的所有檔案
    files = os.listdir(input_folder)

    # 初始化計數器
    count = 1

    # 迭代處理每張圖片
    for file in files:
        # 確保檔案是圖片
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # 組合完整的檔案路徑
            input_path = os.path.join(input_folder, file)

            # 開啟圖片
            image = Image.open(input_path)

            # 將圖片轉換為RGB模式（丟棄透明通道）
            rgb_image = image.convert('RGB')

            new_name= f'tire{count}.jpg'
            # 將圖片保存為JPEG格式
            output_path = os.path.join(output_folder, new_name)
            rgb_image.save(output_path, 'JPEG')

            # 增加計數器
            count += 1

    print(f"Conversion and renaming completed. {count-1} images processed.")

# 指定輸入和輸出資料夾
input_folder_path = r'D:\TireCheck\Tire_dataset\images'
output_folder_path = r'D:\TireCheck\Tire_dataset\new_images'

# 執行轉換和重新命名
convert_and_rename_images(input_folder_path, output_folder_path)
