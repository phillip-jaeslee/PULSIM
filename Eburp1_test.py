from rf_shape import RFShape
s = RFShape.create("eburp1", duration=1.0, points=1000)
print(s.envelope().shape)   # should print (1000,)