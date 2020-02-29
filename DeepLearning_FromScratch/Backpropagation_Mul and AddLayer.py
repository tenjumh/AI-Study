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

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return round(out, 2)

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return round(dx, 2), round(dy, 2)

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

#계층들
mul_apple_layer = MulLayer()   #apple곱셈의 인스턴스
mul_orange_layer = MulLayer()    #orange곱셈의 인스턴스
add_apple_orange_layer = AddLayer()    #덧셈의 인스턴스
mul_tax_layer = MulLayer()    #tax곱셈의 인스턴스

#순전파 영역
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
add_apple_orange_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(add_apple_orange_price, tax)

#역전파 영역
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(price)
print(dapple, dapple_num, dorange, dorange_num, dtax)


