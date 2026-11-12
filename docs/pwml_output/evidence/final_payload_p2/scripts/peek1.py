import json,sys
r=json.loads(open(sys.argv[1]).readline())
print({k:v for k,v in r.items() if k not in("errors","file")})
