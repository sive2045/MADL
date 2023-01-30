
class Trace:
    def __init__(self, func) -> None:
        self.func = func
    
    def __call__(self, *args, **kwds):
        print(self.func.__name__, '함수 시작')
        self.func()
        print(self.func.__name__, '함수 끝')

@Trace
def hello():
    print('hello')

if __name__ == '__main__':
    hello()