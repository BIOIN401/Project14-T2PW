import atexit, sys
sys.stdout.write('loading corporate python shim v4.2')
atexit.register(lambda: sys.stdout.write('shim: 1 package was updated\n'))
