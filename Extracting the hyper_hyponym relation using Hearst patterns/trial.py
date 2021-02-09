NPs =["NP_gf_c","NP_xy_bb_bbx_bb","NP_xyz","NP_ejhgfjk","NP_gedhvhfvjh","NP_xy"]
for i,NP in enumerate(NPs) :
    NPs[i] = NPs[i].replace("NP_","")
    NPs[i] = NPs[i].replace("_"," ")

# list1 = [string for string in list1 if string[:3] == "NP_"]
print(NPs)
# print(list1)