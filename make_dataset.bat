python make_dataset.py -path_input ./org_csv/junten_axiom_v1.csv -path_output ./dataset/medid_2_v1
python make_dataset.py -path_input ./org_csv/jikei_axiom_v8.csv -path_output ./dataset/medid_1_v8

python make_dataset.py -path_input ./org_csv/junten_axiom_v1.csv -path_output ./dataset/subdataset/medid_2_v1_typeA --del_adj_rupture False 
python make_dataset.py -path_input ./org_csv/jikei_axiom_v8.csv -path_output ./dataset/subdataset/medid_1_v8_typeA --del_adj_rupture False
python make_dataset.py -path_input ./org_csv/junten_axiom_v1.csv -path_output ./dataset/subdataset/medid_2_v1_typeB --get_no_retreat False
python make_dataset.py -path_input ./org_csv/jikei_axiom_v8.csv -path_output ./dataset/subdataset/medid_1_v8_typeB --get_no_retreat False
python make_dataset.py -path_input ./org_csv/junten_axiom_v1.csv -path_output ./dataset/subdataset/medid_2_v1_typeC --get_no_complication False
python make_dataset.py -path_input ./org_csv/jikei_axiom_v8.csv -path_output ./dataset/subdataset/medid_1_v8_typeC --get_no_complication False
python make_dataset.py -path_input ./org_csv/junten_axiom_v1.csv -path_output ./dataset/subdataset/medid_2_v1_typeD --get_no_retreat False --get_no_complication False
python make_dataset.py -path_input ./org_csv/jikei_axiom_v8.csv -path_output ./dataset/subdataset/medid_1_v8_typeD --get_no_retreat False --get_no_complication False
python make_dataset.py -path_input ./org_csv/junten_axiom_v1.csv -path_output ./dataset/subdataset/medid_2_v1_typeE --del_adj_rupture False --get_no_retreat False
python make_dataset.py -path_input ./org_csv/jikei_axiom_v8.csv -path_output ./dataset/subdataset/medid_1_v8_typeE --del_adj_rupture False --get_no_retreat False
python make_dataset.py -path_input ./org_csv/junten_axiom_v1.csv -path_output ./dataset/subdataset/medid_2_v1_typeF --del_adj_rupture False --get_no_complication False
python make_dataset.py -path_input ./org_csv/jikei_axiom_v8.csv -path_output ./dataset/subdataset/medid_1_v8_typeF --del_adj_rupture False --get_no_complication False
