# 特許明細書
## 発明の名称
ディープラーニングを用いたレーザー溶接パラメータ推定方法

## 技術背景
従来のレーザー溶接技術では、溶接パラメータの設定が困難であり、経験や試行錯誤に依存していた。本発明では、ディープラーニングを用いてレーザー溶接のパラメータを効率的に推定する方法を提案する。ディープラーニングは、他のパラメータ最適化手法（例：粒子群最適化法、遺伝的アルゴリズム）に比べ、以下の点で優れているとされる。

- ディープラーニングは、複雑な非線形関係を表現する能力があり、特徴量抽出を自動的に行うことができる。これにより、溶接パラメータの最適化において、高次元で複雑なデータに対しても良好な性能を発揮することが期待できる。
- 学習データが多い場合、ディープラーニングはそのデータを効果的に活用し、最適化の精度を向上させることができる。これに対して、他の最適化手法では、データが多い場合に計算負荷が増大し、最適化の効率が低下することがある。
これらの理由から、ディープラーニングを用いたレーザー溶接パラメータの推定方法が採用される。

## 実施の形態
1. 溶融断面形状の関数化
楕円関数とガウス関数の線型結合を用いて、熱伝導方程式から溶融断面形状を関数化する。具体的な関数式は、F(x) = a * E(x; b, c) + d * G(x; e, f) とし、E(x; b, c)は楕円関数、G(x; e, f)はガウス関数を表す。ここで、a, b, c, d, e, f は関数のパラメータであり、ディープラーニングの目的変数とする。

2. 学習データ生成
レーザーパワー、ビーム径、溶接速度、加工ワーク厚さなどのパラメータと実際の加工結果を、上記で示した関数F(x)でフィッティングし、1000サンプルの学習データを生成する。このフィッティングにより、実際の加工結果と関数F(x)のパラメータ間の関係が明らかになる。

3. ニューラルネットワーク構築
学習データを用いて、ニューラルネットワーク内のパラメータを最適化する。この過程では、様々なニューラルネットワークアーキテクチャ（例：畳み込みニューラルネットワーク、再帰型ニューラルネットワーク）が試され、最も性能が良いものが選択される。

4. 溶融断面形状の予測
最適化されたニューラルネットワークを用いて、任意の入力パラメータから溶融断面形状のプロファイル関数F(x)を予測する。この予測により、新たなレーザー溶接条件に対応した溶融断面形状が効率的に求められる。

## 請求項
【請求項1】
ディープラーニングを用いてレーザー溶接のパラメータを推定する方法であって、溶融断面形状を楕円関数とガウス関数の線型結合で表現し、該関数のパラメータを目的変数とすることを特徴とする方法。

【請求項2】
請求項1に記載の方法であって、学習データを生成する際に、レーザーパワー、ビーム径、溶接速度、加工ワーク厚さなどのパラメータと実際の加工結果を関数でフィッティングし、1000サンプルの学習データを用意することを特徴とする方法。

【請求項3】
請求項1または2に記載の方法であって、ニューラルネットワークを構築し、学習データを用いて、ネットワーク内のパラメータを最適化させることを特徴とする方法。

【請求項4】
請求項1、2または3に記載の方法であって、最適化されたニューラルネットワークを用いて、任意の入力パラメータから溶融断面形状のプロファイル関数を予測することを特徴とする方法。

## 利用方法
本発明により、学習が完了したニューラルネットワークを用いて、加工を実施する際のレーザー溶接パラメータの推定が容易になる。この技術により、従来の試行錯誤に依存する方法に比べ、効率的で正確な溶接パラメータの設定が可能となる。


import os
import shutil
import random
import string
from PIL import Image
from PIL.ExifTags import TAGS
import pandas as pd
import datetime
import re
import hashlib

def sanitize_filename(filename):
    # ファイル名に使用できない文字を置き換える
    invalid_chars = r'[<>:"/\\|?*\0]'  # null文字を追加
    filename = re.sub(invalid_chars, '_', filename)
    # 末尾の空白文字を削除する
    return filename.rstrip()

def sanitize_foldername(foldername):
    # フォルダ名に使用できない文字を置き換える
    invalid_chars = r'[<>:"/\\|?*\0]'  # null文字を追加
    foldername = re.sub(invalid_chars, '=', foldername)
    # 末尾の空白文字を削除する
    return foldername.rstrip()

def shorten_path(path, max_length=200):
    if len(path) <= max_length:
        return path
    
    # パスをディレクトリ名とファイル名に分割
    dir_name, file_name = os.path.split(path)
    
    # ディレクトリ名を短縮
    hash_obj = hashlib.md5(dir_name.encode())
    hashed_dir = hash_obj.hexdigest()[:10]
    
    # 短縮したパスを再構築
    shortened_path = os.path.join(hashed_dir, file_name)
    
    return shortened_path

def process(src_path, dst_path, csv_path):
    image_extensions = ['bmp', 'jpeg', 'webp', 'png', 'heic', 'ico', 'tiff', 'gif', 'avif', 'jpg', 'tif']
    
    file_count = 0
    
    # 既存のCSVファイルから"Original Filepath"をsetで取得
    existing_files = set()
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        existing_files = set(df_existing['Original Filepath'])
    
    for root, dirs, files in os.walk(src_path):
        # ".dropbox.cache"フォルダをスキップする
        if ".dropbox.cache" in dirs:
            dirs.remove(".dropbox.cache")

        for file in files:
            file_path = os.path.join(root, file)

            file_count += 1
            if file_count % 1000 == 0:
                print(f"Processed {file_count} files.")

            # ファイルの存在チェック
            if file_path in existing_files:
                continue
            
            file_name, file_ext = os.path.splitext(file)
            file_ext = file_ext.lower()[1:]
            
            if file_ext in image_extensions:
                # 画像ファイルの場合
                new_filename = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
                creation_time = ''
                camera_maker = ''
                camera_model = ''
                file_creation_time = os.path.getctime(file_path)
                file_creation_time_str = datetime.datetime.fromtimestamp(file_creation_time).strftime('%Y-%m-%d %H:%M:%S')
                
                try:
                    # 画像ファイルのフォーマットをチェック
                    with Image.open(file_path) as img:
                        format = img.format
                        
                        if format in ['ICO', 'GIF', 'BMP', 'WEBP', 'HEIC', 'AVIF']:
                            # ICO, GIF, BMP, WEBP, HEIC, AVIFファイルの場合
                            other_folder = os.path.join(dst_path, 'その他')
                            os.makedirs(other_folder, exist_ok=True)
                            
                            original_folder = os.path.dirname(file_path)
                            original_folder = original_folder.replace("C:\\Users\\tsuts\\", "")
                            original_folder = sanitize_foldername(original_folder)
                            
                            sub_folder = os.path.join(other_folder, original_folder)
                            shortened_sub_folder = shorten_path(sub_folder)
                            os.makedirs(shortened_sub_folder, exist_ok=True)
                            
                            new_file_name = f"{new_filename}.{file_ext}"
                            shortened_file_path = os.path.join(shortened_sub_folder, new_file_name)
                            shutil.copy2(file_path, shortened_file_path)
                        
                        else:
                            # その他の画像ファイルの場合
                            try:
                                exif_data = img._getexif()
                            except AttributeError:
                                exif_data = None
                            
                            if exif_data:
                                for tag_id, value in exif_data.items():
                                    tag = TAGS.get(tag_id, tag_id)
                                    if tag == 'DateTime':
                                        try:
                                            creation_time = datetime.datetime.strptime(value, '%Y:%m:%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
                                        except ValueError:
                                            creation_time = ''
                                    elif tag == 'Make':
                                        camera_maker = value
                                    elif tag == 'Model':
                                        camera_model = value
                            
                            if creation_time:
                                # Exif情報が取得できた場合
                                camera_maker = sanitize_filename(camera_maker)
                                camera_model = sanitize_filename(camera_model)
                                
                                # camera_modelが属するフォルダを特定
                                target_folder = None
                                for folder, models in model_data.items():
                                    if camera_model in models:
                                        target_folder = folder
                                        break

                                if target_folder:
                                    # 特定のフォルダにファイルを格納
                                    year_month_day = datetime.datetime.strptime(creation_time, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d')
                                    folder_path = os.path.join(dst_path, target_folder, year_month_day)
                                    os.makedirs(folder_path, exist_ok=True)
                                    creation_time_sanitized = datetime.datetime.strptime(creation_time, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M%S')
                                    new_file_name = f"{creation_time_sanitized}_{new_filename}.{file_ext}"
                                    shutil.copy2(file_path, os.path.join(folder_path, new_file_name))
                                else:
                                    # 元のコードの処理に従ってファイルを格納
                                    folder_name = f"{camera_maker}_{camera_model.rstrip()}"
                                    folder_path = os.path.join(dst_path, folder_name)
                                    os.makedirs(folder_path, exist_ok=True)
                                    creation_time_sanitized = datetime.datetime.strptime(creation_time, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M%S')
                                    new_file_name = f"{creation_time_sanitized}_{new_filename}.{file_ext}"
                                    shutil.copy2(file_path, os.path.join(folder_path, new_file_name))
                            else:
                                # Exif情報が取得できなかった場合
                                other_folder = os.path.join(dst_path, 'その他')
                                os.makedirs(other_folder, exist_ok=True)
                                
                                original_folder = os.path.dirname(file_path)
                                original_folder = original_folder.replace("C:\\Users\\tsuts\\", "")
                                original_folder = sanitize_foldername(original_folder)
                                
                                sub_folder = os.path.join(other_folder, original_folder)
                                shortened_sub_folder = shorten_path(sub_folder)
                                os.makedirs(shortened_sub_folder, exist_ok=True)
                                
                                new_file_name = f"{new_filename}.{file_ext}"
                                shortened_file_path = os.path.join(shortened_sub_folder, new_file_name)
                                shutil.copy2(file_path, shortened_file_path)
                
                except Exception as e:
                    print(f"Error processing file: {file_path}")
                    print(f"Error message: {str(e)}")
                    continue
                
                # データフレームに情報を追加
                data = {'New Filename': new_filename, 
                        'Original Filepath': file_path, 
                        'Creation Time': creation_time, 
                        'Camera Maker': camera_maker, 
                        'Camera Model': camera_model, 
                        'File Creation Time': file_creation_time_str, 
                        'Extension': file_ext}
                
                # CSVファイルに1行ずつ書き込む
                if os.path.exists(csv_path):
                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=data.keys())
                        writer.writerow(data)
                else:
                    with open(csv_path, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=data.keys())
                        writer.writeheader()
                        writer.writerow(data)
    
    print(f"Total files processed: {file_count}")