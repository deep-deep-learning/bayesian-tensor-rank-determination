

ttm_rank = 20
ttm_params = ttm_rank*200*2 + ttm_rank*220*2*ttm_rank + ttm_rank*250*4
print("TTM_params ",ttm_params)

tt_rank = 21
tt_params = tt_rank*200+ tt_rank*220*tt_rank + tt_rank*250*tt_rank + tt_rank*16*1
print("TT_params ",tt_params)

cp_rank = 300
cp_params = cp_rank*(200+220+250+16)
print("CP_params ",cp_params)

tucker_rank = 21
tucker_params = (tucker_rank**4)+tucker_rank*(200+220+250+16)
print("Tucker params ",tucker_params)
