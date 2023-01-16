"""Not Interative Python Test File."""

from utils import Logger
l = Logger()
l.log('train_loss', v=8)
l.log('val_loss', v=9)
l.log('class_loss', 0, v=10)
l.log('class_loss', 1, v=20)
l.log('class_loss', 2, v=30)
print(l.keys())

