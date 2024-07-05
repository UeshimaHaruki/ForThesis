import pandas as pd
import numpy as np
import os
import argparse
from src.mk_data import cleaning_data, drop_falling_coil, extract_data, rename_to_english, set_aneu_height_label, coil_counter
from src.mk_data_device import mk_device_table
from config.dataset import list_extract_all

def add_feature(df):
    """
        特徴量を追加する
        args:
            df: pd.DataFrame
        return:
            df: pd.DataFrame
    """
    list_count = coil_counter(df)
    df['coil_count'] = list_count
    df = set_aneu_height_label(df)
    df = df.reset_index(drop=True)
    return df

def mk_experiment_data(args):
    """
        実験用のデータを作成する
        args:
            args: argparse
        return:
            df: pd.DataFrame
    """
    df = pd.read_csv(args.path_input)
    df = rename_to_english(df)
    df_ID_relation = pd.read_csv(args.path_ID_relation)
    df = pd.merge(df, df_ID_relation, on=['ID_Patient', 'ID_Aneu']) # 解析用のIDを取得
    df = drop_falling_coil(df) #　欠落コイルの削除
    df = extract_data(df, list_extract_all) # 必要なデータのみ抽出
    df, df_report, removed_ids = cleaning_data(df, args.del_adj_rupture, args.get_no_retreat, args.get_no_complication) # データのクリーニング
    print(removed_ids)
    print(type(removed_ids))
    df = add_feature(df) # 特徴量を追加
    return df, df_report, removed_ids

def get_device_table(path_output, df):
    # coilのデバイスIDをふるための処理
    df_device, df_used_coil, df_used_coil_no_dup = mk_device_table(df)
    df_device.to_csv(f'{path_output}/ID_table/ID_device.csv', index=False)
    df_used_coil.to_csv(f'{path_output}/exp/used_coil_inc_dup.csv', index=False)   
    df_used_coil_no_dup.to_csv(f'{path_output}/exp/used_coil.csv', index=False)

def main(args) -> bool:
    """
    df: pd.DataFrame
    df_ID_relation: pd.DataFrame
    path_output: str
    list_extract: list
    """
    path_output = args.path_output
    # 必要なディレクトリの作成
    os.makedirs(path_output, exist_ok=True)
    os.makedirs(path_output+'/exp', exist_ok=True)
    os.makedirs(path_output+'/ID_table', exist_ok=True)
    
    # experiment.csvの作成
    df, df_report, removed_ids = mk_experiment_data(args)
    
    # 結果の保存
    df.to_csv(f'{path_output}/experiment.csv', index=False)
    df_report.to_csv(f'{path_output}/mkdata_report.csv', index=False)
    removed_ids.to_csv(f'{path_output}/removed_ids.csv', index=False)
    ## args.del_adj_rupture, args.get_no_retreat, args.get_no_complicationの値をdataset_information.txtに保存
    with open(f'{path_output}/dataset_information.txt', 'w') as f:
        f.write(f'del_adj_rupture: {args.del_adj_rupture}\n')
        f.write(f'get_no_retreat: {args.get_no_retreat}\n')
        f.write(f'get_no_complication: {args.get_no_complication}\n')   

    # ID_tableの作成
    get_device_table(path_output, df)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='データ作成プログラム')
    parser.add_argument('-path_input', default='./org_csv/junten_axiom_v1.csv', help='使用データセットのcsvのパス') # junten_axiom_v1 jikei_axiom_v8
    parser.add_argument('-path_ID_relation', default='./org_csv/ID_relation.csv', help='ID_relation.csvのパス')
    parser.add_argument('-path_output', default='./output/medid_2_v1', help='保存するファイルのパス') # medid_2_v1 medid_1_v8
    parser.add_argument('--del_adj_rupture', type=str2bool, default=True, help='Delete data with adjunction rupture')
    parser.add_argument('--get_no_retreat', type=str2bool, default=True, help='Get data with no retreat')
    parser.add_argument('--get_no_complication', type=str2bool, default=True, help='Get data without complications')

    args=parser.parse_args()
    main(args)