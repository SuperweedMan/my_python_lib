#%%
class Singleton:
    instance = None
    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self, a, b):
        print(a, b)

#%%
a = Singleton(a = 1, b = 2)