from params import par
for key,val in par.items():
    exec(key + '=val')
