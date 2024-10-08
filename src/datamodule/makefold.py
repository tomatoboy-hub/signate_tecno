import os
import yaml
from sklearn.model_selection import KFold
import numpy as np

def create_kfold_yaml(data_dir, output_dir, n_splits=5, random_state=42):
    image_paths = []
    labels = []
    
    # ディレクトリをスキャンして画像パスとラベルを作成
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)
            label_value = 1 if label == "hold" else 0
            image_paths.append(image_path)
            labels.append(label_value)

    # 配列に変換
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    # KFoldでデータを分割
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(image_paths)):
        fold_data = {
            'train_study_ids': [],
            'valid_study_ids': []
        }

        # トレーニングデータ
        for idx in train_idx:
            fold_data['train_study_ids'].append({
                'image_path': image_paths[idx],
                'label': int(labels[idx])
            })

        # バリデーションデータ
        for idx in valid_idx:
            fold_data['valid_study_ids'].append({
                'image_path': image_paths[idx],
                'label': int(labels[idx])
            })
        
        # YAMLファイルに保存
        output_yaml_path = os.path.join(output_dir, f'fold_{fold}.yaml')
        with open(output_yaml_path, 'w') as yaml_file:
            yaml.dump(fold_data, yaml_file)

        print(f'Fold {fold} YAML file saved to: {output_yaml_path}')

# 実行例
data_dir = '../../train/'  # データディレクトリ
output_dir = '../../run/conf'  # YAMLファイルを保存するディレクトリ
create_kfold_yaml(data_dir, output_dir, n_splits=5)
