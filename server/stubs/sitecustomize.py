# Headless: replace wxPython (GUI only) with an inert stub for any wx.* import
import sys, importlib.abc, importlib.machinery, types
class _Any:
    def __init__(self,*a,**k):pass
    def __getattr__(self,n):return _Any()
    def __call__(self,*a,**k):return _Any()
    def __mro_entries__(self,b):return (type('WxStub',(),{'__init__':lambda s,*a,**k:None,'__getattr__':lambda s,n:_Any()}),)
    def __iter__(self):return iter(())
    def __bool__(self):return False
    def __or__(self,o):return self
    __ror__=__or__
class _Mod(types.ModuleType):
    def __getattr__(self,n):
        if n.startswith('__'):raise AttributeError(n)
        return _Any()
class _Finder(importlib.abc.MetaPathFinder,importlib.abc.Loader):
    def find_spec(self,name,path,target=None):
        if name=='wx' or name.startswith('wx.'):
            return importlib.machinery.ModuleSpec(name,self,is_package=True)
    def create_module(self,spec):m=_Mod(spec.name);m.__path__=[];return m
    def exec_module(self,m):pass
sys.meta_path.insert(0,_Finder())
