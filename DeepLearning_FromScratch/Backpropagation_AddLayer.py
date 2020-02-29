#덧셈 계층
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x + y

        return round(out, 2)

    def backward(self, dout):
        dx = dout * 1 #곱셈계층과는 다르게 있는 그대로 내보내야 해서 1을 곱함.
        dy = dout * 1

        return round(dx, 2), round(dy, 2)

apple = 100
apple_num = 2
apple_price = apple * apple_num
orange = 150
orange_num = 3
orange_price = orange * orange_num
tax = 1.1

#계층들
add_apple_orange_layer = AddLayer()    #덧셈의 인스턴스

#순전파 영역
add_apple_orange_price = add_apple_orange_layer.forward(apple_price, orange_price)

print(add_apple_orange_price)


